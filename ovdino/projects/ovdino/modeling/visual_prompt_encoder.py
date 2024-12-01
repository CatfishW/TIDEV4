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
        height, width = last_layer_feature.shape[-2:]

        # Initialize batch-specific averaged features
        batched_averaged_features = []

        for batch_idx, data in enumerate(batched_input):
            instances = data['instances']  # Corrected 'instance' to 'instances'
            gt_boxes = instances.gt_boxes.tensor.clone()  # Use the tensor of bounding boxes
            gt_classes = instances.gt_classes

            # Compute scaling factors
            scale_x = width / data['instances'].image_size[1]
            scale_y = height / data['instances'].image_size[0]

            # Scale the box coordinates according to the feature map resolution
            gt_boxes[:, 0] *= scale_x  # x1
            gt_boxes[:, 2] *= scale_x  # x2
            gt_boxes[:, 1] *= scale_y  # y1
            gt_boxes[:, 3] *= scale_y  # y2

            # Initialize class-specific feature maps for this batch
            class_feature_map_dict = {cls: [] for cls in range(80)}

            # Extract feature map inside each scaled gt_box
            for i, box in enumerate(gt_boxes):
                x1, y1, x2, y2 = box.int()

                feature_map = last_layer_feature[batch_idx:batch_idx+1, :, y1:y2, x1:x2]

                if feature_map.numel() > 0:  # Ensure there are elements to pool
                    pooled_feature = F.adaptive_avg_pool2d(feature_map, (1, 1)).squeeze()
                    class_feature_map_dict[gt_classes[i].item()].append(pooled_feature)

            # Average feature maps that belong to the same ground truth class
            averaged_features = [torch.zeros(last_layer_feature.shape[1], device=last_layer_feature.device) for _ in range(80)]

            for cls, maps in class_feature_map_dict.items():
                if maps:
                    averaged_features[cls] = torch.stack(maps).mean(dim=0)
                else:
                    # Sample negative feature map from last_layer_feature
                    random_y = torch.randint(0, height, (1,))
                    random_x = torch.randint(0, width, (1,))
                    negative_sample = last_layer_feature[batch_idx:batch_idx+1, :, random_y, random_x].squeeze()
                    averaged_features[cls] = negative_sample

            # Add batch-specific averaged features to the list
            batched_averaged_features.append(torch.stack(averaged_features))

        # Return batched tensor of averaged features
        return torch.stack(batched_averaged_features)

