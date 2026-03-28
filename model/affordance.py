import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pointnet import PointNet2MSG

class Affordance3DBranch(nn.Module):

    def __init__(self, input_dim=6, hidden_dim=512, num_points=100):
        super(Affordance3DBranch, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_points = num_points
        self.pointnet_backbone = PointNet2MSG(
            input_dim=input_dim,
            output_dim=hidden_dim,
            num_points=num_points
        )
        self.affordance_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_points)  
        )
        
    def forward(self, point_clouds):
        global_features = self.pointnet_backbone(point_clouds)
        affordance_logits = self.affordance_decoder(global_features)
        affordance_scores = torch.sigmoid(affordance_logits)
        
        return affordance_scores

class AffordanceConditioner(nn.Module):
    def __init__(self, affordance_dim=100, latent_dim=512):
        super(AffordanceConditioner, self).__init__()
        self.affordance_proj = nn.Sequential(
            nn.Linear(affordance_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.temporal_encoder = nn.Linear(1, 256)
        
    def forward(self, affordance_scores, temporal_weights=None):
        B, T, N = affordance_scores.shape
        affordance_flat = affordance_scores.view(B*T, N)
        affordance_features = self.affordance_proj(affordance_flat)  # [B*T, latent_dim]
        affordance_features = affordance_features.view(B, T, -1)  # [B, T, latent_dim]
        if temporal_weights is not None:
            temporal_encoding = self.temporal_encoder(temporal_weights.unsqueeze(-1))  # [B, T, latent_dim]
            affordance_features = affordance_features * temporal_encoding
        
        return affordance_features