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
        """Optimized processing of feature map."""
        last_layer_feature = features[-1]

        # Resize the feature map to a fixed resolution (e.g., 100x100)
        resized_feature = F.interpolate(last_layer_feature, size=(40,40), mode="bilinear", align_corners=False)
        resized_height, resized_width = resized_feature.shape[-2:]

        # Precompute scales for box resizing
        scales = [
            (resized_width / data["instances"].image_size[1], resized_height / data["instances"].image_size[0])
            for data in batched_input
        ]

        # Precompute and store resized GT boxes and class assignments
        all_gt_boxes = []
        all_gt_classes = []

        for batch_idx, data in enumerate(batched_input):
            instances = data["instances"]
            gt_boxes = instances.gt_boxes.tensor.clone()
            gt_classes = instances.gt_classes

            # Scale boxes
            scale_x, scale_y = scales[batch_idx]
            gt_boxes[:, [0, 2]] *= scale_x  # x1, x2
            gt_boxes[:, [1, 3]] *= scale_y  # y1, y2

            all_gt_boxes.append(gt_boxes)
            all_gt_classes.append(gt_classes)

        # Prepare batched feature map
        batched_averaged_features = []

        for batch_idx in range(len(batched_input)):
            gt_boxes = all_gt_boxes[batch_idx]
            gt_classes = all_gt_classes[batch_idx]

            # Initialize class-specific feature map dictionary
            class_feature_map_dict = {cls: [] for cls in range(599)}

            for i, box in enumerate(gt_boxes):
                # Round and clamp box coordinates
                x1, y1, x2, y2 = torch.round(box).long()
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, resized_width), min(y2, resized_height)

                if x1 < x2 and y1 < y2:
                    feature_map = resized_feature[batch_idx:batch_idx + 1, :, y1:y2, x1:x2]
                    if feature_map.numel() > 0:
                        pooled_feature = F.adaptive_avg_pool2d(feature_map, (1, 1)).squeeze()
                        class_feature_map_dict[gt_classes[i].item()].append(pooled_feature)

            # Compute averaged features per class
            averaged_features = torch.zeros((599, resized_feature.shape[1]), device=resized_feature.device)
            for cls, maps in class_feature_map_dict.items():
                if maps:
                    averaged_features[cls] = torch.stack(maps).mean(dim=0)
                elif self.training:
                    # Sample negative features
                    random_y = torch.randint(0, resized_height, (1,))
                    random_x = torch.randint(0, resized_width, (1,))
                    negative_sample = resized_feature[batch_idx:batch_idx + 1, :, random_y, random_x].squeeze()
                    averaged_features[cls] = negative_sample

            batched_averaged_features.append(averaged_features)

        # Stack results for all batches
        return torch.stack(batched_averaged_features)




