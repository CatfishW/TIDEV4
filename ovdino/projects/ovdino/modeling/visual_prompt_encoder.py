import torch
import torch.nn as nn
from detrex.layers import (
    FFN,
    MLP,
    BaseTransformerLayer,
    MultiheadAttention,
    MultiScaleDeformableAttention,
)
from .positional_encoding import PositionEmbeddingRandom
import torch.nn.functional as F
class VisualPromptEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0
        # self.pe_layer = PositionEmbeddingRandom(256 // 2)
        # self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        # point_embeddings = [nn.Embedding(1, 256) for i in range(self.num_point_embeddings)]
        # self.point_embeddings = nn.ModuleList(point_embeddings)
        # #===================================
        # self.content_embedding = nn.Embedding(1, 256)
        # #self.cross_attention_vp = nn.MultiheadAttention(self.hidden_dim, self.num_heads, dropout=self.dropout)
        # self.cross_attention_vp = MultiScaleDeformableAttention(
        #             embed_dim=256,
        #             num_heads=8,
        #             dropout=0,
        #             batch_first=True,
        #             num_levels=4,
        #         )
        # self.cross_attention_vp_dropout = (
        #     nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()
        # )
        # self.cross_attention_vp_norm = nn.LayerNorm(256)
        # self.self_attn = nn.MultiheadAttention(256,8, dropout=0,batch_first=True)

        # # ffn
        # self.ffn = FFN(
        #     embed_dim=256,
        #     feedforward_dim=1024,
        #     num_fcs=2,
        #     ffn_drop=0.0,
        #     add_identity=True,
        # )

        # self.cls_token = nn.Parameter(torch.zeros(1, 1,256))
    def _embed_boxes(self, boxes: torch.Tensor,input_image_size) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of dpixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords,input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        #average the two corners
        corner_embedding = corner_embedding.mean(dim=1)
        return corner_embedding
    
    def _process_feature_map(self, features, batched_input) -> torch.Tensor:
        """Processes feature map."""
        last_layer_feature = features[-1]

        # Resize the feature map to a fixed resolution (e.g., 640x640)
        resized_feature = F.interpolate(last_layer_feature, size=(640, 640), mode="bilinear", align_corners=False)
        resized_height, resized_width = resized_feature.shape[-2:]

        # Initialize batch-specific averaged features
        batched_averaged_features = []

        for batch_idx, data in enumerate(batched_input):
            instances = data['instances']
            gt_boxes = instances.gt_boxes.tensor.clone()  # Use the tensor of bounding boxes
            gt_classes = instances.gt_classes
            original_image_height, original_image_width = data['instances'].image_size

            # Scale ground-truth boxes to the resized feature map resolution
            scale_x = resized_width / original_image_width
            scale_y = resized_height / original_image_height

            gt_boxes[:, 0] *= scale_x  # x1
            gt_boxes[:, 2] *= scale_x  # x2
            gt_boxes[:, 1] *= scale_y  # y1
            gt_boxes[:, 3] *= scale_y  # y2

            # Create a mask for positive regions
            positive_mask = torch.zeros((resized_height, resized_width), dtype=torch.bool, device=resized_feature.device)
            for box in gt_boxes:
                x1, y1, x2, y2 = torch.round(box).long()
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, resized_width), min(y2, resized_height)
                positive_mask[y1:y2, x1:x2] = True

            # Initialize class-specific feature maps for this batch
            class_feature_map_dict = {cls: [] for cls in range(80)}

            # Extract feature map inside each scaled gt_box
            for i, box in enumerate(gt_boxes):
                # Round coordinates to integers for slicing
                x1, y1, x2, y2 = torch.round(box).long()

                # Ensure box coordinates are within bounds
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, resized_width), min(y2, resized_height)

                if x1 < x2 and y1 < y2:  # Ensure the box has a valid area
                    feature_map = resized_feature[batch_idx:batch_idx+1, :, y1:y2, x1:x2]

                    if feature_map.numel() > 0:  # Ensure there are elements to pool
                        pooled_feature = F.adaptive_avg_pool2d(feature_map, (1, 1)).squeeze()
                        class_feature_map_dict[gt_classes[i].item()].append(pooled_feature)

            # Average feature maps that belong to the same ground truth class
            averaged_features = [torch.zeros(resized_feature.shape[1], device=resized_feature.device) for _ in range(80)]

            for cls, maps in class_feature_map_dict.items():
                if maps:
                    averaged_features[cls] = torch.stack(maps).mean(dim=0)
                else:
                    if self.training:
                        # Sample negative feature map from regions outside ground-truth boxes
                        while True:
                            # Randomly choose a box size between 5x5 and 30x30
                            negative_height = torch.randint(5, 31, (1,)).item()  # Random height from 5 to 30
                            negative_width = torch.randint(5, 31, (1,)).item()   # Random width from 5 to 30

                            # Ensure the sampled box fits within the image dimensions
                            random_y = torch.randint(0, resized_height - negative_height + 1, (1,)).item()
                            random_x = torch.randint(0, resized_width - negative_width + 1, (1,)).item()

                            # Check if the randomly chosen area overlaps with any positive regions
                            negative_box_mask = torch.zeros((resized_height, resized_width), dtype=torch.bool, device=resized_feature.device)
                            negative_box_mask[random_y:random_y+negative_height, random_x:random_x+negative_width] = True

                            if not (positive_mask & negative_box_mask).any():  # No overlap
                                break

                        # Extract negative feature map
                        negative_sample = resized_feature[batch_idx:batch_idx+1, :, random_y:random_y+negative_height, random_x:random_x+negative_width].squeeze()

                        # Average the negative feature map to match the required output shape
                        averaged_features[cls] = F.adaptive_avg_pool2d(negative_sample.unsqueeze(0), (1, 1)).squeeze()

            # Add batch-specific averaged features to the list
            batched_averaged_features.append(torch.stack(averaged_features))

        # Return batched tensor of averaged features
        return torch.stack(batched_averaged_features)




