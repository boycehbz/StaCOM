import torch
from torch import nn


class contact_point_generator(nn.Module):

    def __init__(
        self,
        smpl=None,
        num_joints=21,
        frame_length=128,
        bps_dim=1024,
        point_feat_dim=128,
        point_input_dim=7,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        max_people=2,
        num_hands=2,
    ):
        super().__init__()
        self.max_people = max_people
        self.num_hands = num_hands
        self.num_queries = max_people * num_hands

        self.pose_proj = nn.Sequential(
            nn.LayerNorm(12),
            nn.Linear(12, hidden_dim),
            nn.GELU(),
        )
        self.bps_proj = nn.Sequential(
            nn.LayerNorm(bps_dim),
            nn.Linear(bps_dim, hidden_dim),
            nn.GELU(),
        )

        self.point_mlp = nn.Sequential(
            nn.Linear(point_input_dim, point_feat_dim),
            nn.GELU(),
            nn.Linear(point_feat_dim, point_feat_dim),
            nn.GELU(),
        )
        self.affordance_embed = nn.Sequential(
            nn.Linear(1, point_feat_dim),
            nn.GELU(),
        )
        self.affordance_context = nn.Sequential(
            nn.Linear(point_feat_dim, hidden_dim),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu',
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.query_embed = nn.Parameter(torch.randn(self.num_queries, hidden_dim) * 0.02)
        self.query_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.point_head = nn.Linear(hidden_dim, 3)
        self.valid_head = nn.Linear(hidden_dim, 1)

    def _extract_object_pose_feature(self, obj_pose):
        rot = obj_pose[..., :3, :3].reshape(*obj_pose.shape[:2], 9)
        trans = obj_pose[..., :3, 3]
        return torch.cat([rot, trans], dim=-1)

    def forward(self, data):
        obj_bps = data['obj_bps']
        obj_pose = data['obj_pose']
        obj_points = data['obj_points']

        B, T = obj_bps.shape[:2]
        N = obj_points.shape[2]

        pose_feat = self.pose_proj(self._extract_object_pose_feature(obj_pose))
        bps_feat = self.bps_proj(obj_bps)
        frame_cond = pose_feat + bps_feat

        point_feat = self.point_mlp(obj_points)
        if obj_points.shape[-1] >= 7:
            affordance_prob = obj_points[..., 6].clamp(0.0, 1.0)
        else:
            affordance_prob = torch.zeros((B, T, N), device=obj_points.device, dtype=obj_points.dtype)

        affordance_feat = self.affordance_embed(affordance_prob.unsqueeze(-1))
        point_feat = point_feat + affordance_feat

        point_weights = affordance_prob / (affordance_prob.sum(dim=2, keepdim=True) + 1e-6)
        affordance_ctx = torch.sum(point_feat * point_weights.unsqueeze(-1), dim=2)
        affordance_ctx = self.affordance_context(affordance_ctx)

        frame_tokens = frame_cond + affordance_ctx
        encoded = self.temporal_encoder(frame_tokens)

        query = self.query_embed.view(1, 1, self.num_queries, -1).expand(B, T, -1, -1)
        token_expand = encoded.unsqueeze(2).expand(-1, -1, self.num_queries, -1)
        query_feat = self.query_mlp(torch.cat([token_expand, query], dim=-1))

        pred_points = self.point_head(query_feat).view(B, T, self.max_people, self.num_hands, 3)
        pred_logits = self.valid_head(query_feat).view(B, T, self.max_people, self.num_hands)

        return {
            'pred_contact_points': pred_points,
            'pred_contact_logits': pred_logits,
            'pred_contact_valid': torch.sigmoid(pred_logits),
            'cond_affordance': affordance_prob,
        }