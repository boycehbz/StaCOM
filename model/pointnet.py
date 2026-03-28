import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.zeros(B, dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

class PointNet2MSG(nn.Module):
    def __init__(self, input_dim=6, output_dim=512, num_points=200):
        super(PointNet2MSG, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_points = num_points
        
        # Set abstraction layers
        # self.sa1 = PointNetSetAbstractionMsg(64, [0.1, 0.2, 0.4], [16, 32, 128], 
        #                                    input_dim, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # self.sa2 = PointNetSetAbstractionMsg(16, [0.2, 0.4, 0.8], [32, 64, 128], 
        #                                    320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])

        self.sa1 = PointNetSetAbstractionMsg(32, [0.1, 0.2, 0.4], [8, 16, 32], 
                                   input_dim, [[16, 16, 32], [32, 32, 64], [32, 48, 64]])
        self.sa2 = PointNetSetAbstractionMsg(8, [0.2, 0.4, 0.8], [8, 16, 32], 
                                        160, [[32, 32, 64], [64, 64, 128], [64, 64, 128]])
        
        self.sa3 = PointNetSetAbstraction(None, None, None, 320 + 3, [256, 512, 1024], True)
    
        
        # Feature propagation layers for producing dense features
        self.fp3 = PointNetFeaturePropagation(1344, [256, 256])  # 320 + 1024 = 1344
        self.fp2 = PointNetFeaturePropagation(416, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + input_dim, [128, 128, 128])
        
        # Final feature extraction
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, output_dim, 1)

        self.global_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, xyz_points):
        B, N, _ = xyz_points.shape
        
        # Split xyz and features
        xyz = xyz_points[:, :, :3].contiguous()  # [B, N, 3]
        # points = xyz_points[:, :, 3:].contiguous()  # [B, N, 3] (normals)
        points = xyz_points.contiguous()  
        
        # Transpose for PointNet++ convention: [B, C, N]
        xyz = xyz.permute(0, 2, 1)  # [B, 3, N]
        points = points.permute(0, 2, 1)  # [B, 3, N]
        
        # Set abstraction layers
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Feature propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, points, l1_points)
        
        # Final feature extraction
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = self.drop1(feat)
        feat = self.conv2(feat)  # [B, output_dim, N]
        # print(feat.shape)
        # Global pooling to get single feature vector per point cloud
        global_feat = self.global_pool(feat).squeeze(-1)  # [B, output_dim]
        
        return global_feat

class AffordancePredictor(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256):
        super(AffordancePredictor, self).__init__()
        self.input_dim = input_dim
        
        # Point-wise feature extraction
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, hidden_dim, 1)
        
        # Affordance prediction head
        self.affordance_head = nn.Sequential(
            nn.Conv1d(hidden_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, 1)  # Single affordance score per point
        )
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, xyz_points):
        # Transpose for conv1d: [B, C, N]
        points = xyz_points.permute(0, 2, 1)  # [B, 6, N]
        
        # Point-wise feature extraction
        x = F.relu(self.bn1(self.conv1(points)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Affordance prediction
        affordance_logits = self.affordance_head(x)  # [B, 1, N]
        affordance_scores = torch.sigmoid(affordance_logits.squeeze(1))  # [B, N]
        
        return affordance_scores

class ContactToAffordanceConverter(nn.Module):
    def __init__(self, contact_threshold=0.05, affordance_radius=0.1):
        super(ContactToAffordanceConverter, self).__init__()
        self.contact_threshold = contact_threshold
        self.affordance_radius = affordance_radius
        
    def generate_affordance_labels(self, joints, obj_vertices):
        B, T, N, _ = obj_vertices.shape
        num_joints = joints.shape[2]
        hand_joints = joints[:, :, [20, 21]]
        
        affordance_labels = torch.zeros(B, T, N, device=joints.device)
        
        for b in range(B):
            for t in range(T):
                for n in range(N):
                    obj_point = obj_vertices[b, t, n]  # [3]
                    distances = torch.norm(hand_joints[b, t] - obj_point.unsqueeze(0), dim=-1)  # [2]
                    min_distance = torch.min(distances)
                    if min_distance < self.contact_threshold:
                        affordance_labels[b, t, n] = 1.0
                    elif min_distance < self.affordance_radius:
                        affordance_labels[b, t, n] = torch.exp(-min_distance / self.affordance_radius)
                    
        return affordance_labels