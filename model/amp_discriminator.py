import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
    
class AMPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.num_joints = 21  
        
        self.D_conv1 = nn.Conv2d(9, 32, kernel_size=1)
        self.D_conv2 = nn.Conv2d(32, 32, kernel_size=1)
        pose_out = []
        for i in range(self.num_joints):
            pose_out.append(nn.Linear(32, 1))
        self.pose_out = nn.ModuleList(pose_out)
    
        self.D_alljoints_fc1 = nn.Linear(32*self.num_joints, 1024)
        self.D_alljoints_fc2 = nn.Linear(1024, 1024)
        self.D_alljoints_out = nn.Linear(1024, 1)
    
    def forward(self, poses, betas=None):

        poses = poses.reshape(-1, self.num_joints, 1, 9)
        bn = poses.shape[0]

        poses = poses.permute(0, 3, 1, 2).contiguous()
    
        poses = F.relu(self.D_conv1(poses))
        poses = F.relu(self.D_conv2(poses))
        

        poses_out = []
        for i in range(self.num_joints):
            poses_out.append(self.pose_out[i](poses[:, :, i, 0]))
        poses_out = torch.cat(poses_out, dim=1)  # [B, 21]
        # [B, 1]

        poses_flat = poses.reshape(bn, -1)
        poses_all = F.relu(self.D_alljoints_fc1(poses_flat))
        poses_all = F.relu(self.D_alljoints_fc2(poses_all))
        poses_all_out = self.D_alljoints_out(poses_all)  # [B, 1]
        
        disc_out = torch.cat([poses_out, poses_all_out], dim=1)
        return disc_out

class AMPTrainer:
    def __init__(self, device='cuda:0', discriminator_ckpt='', load_pretrained=False):
        self.device = device
        self.discriminator = AMPDiscriminator().to(device)
        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=1e-4,
            betas=(0.5, 0.999)
        )
        self.uses_external_real = False

        joint_weights = torch.ones(22, device=self.device)
        for idx in [16, 17, 18, 19]:
            joint_weights[idx] = 1.5
        for idx in [20, 21]:
            joint_weights[idx] = 2.0
        joint_weights[-1] = 1.0
        self.joint_weights = joint_weights

        if load_pretrained and discriminator_ckpt:
            try:
                self.load_discriminator(discriminator_ckpt, load_optimizer=False)
            except FileNotFoundError as err:
                print(f"[AMP] {err}. Continuing with randomly initialized discriminator.")
            except RuntimeError as err:
                print(f"[AMP] Failed to load discriminator weights: {err}. Using random initialization instead.")
        
    def _get_real_batch(self, batch_size, real_poses):
        if real_poses is None:
            raise ValueError('Real poses must be provided for AMP training.')

        return real_poses.to(self.device)

    def _weighted_logits(self, pred):
        weights = self.joint_weights
        norm = weights.sum()
        return (pred * weights).sum(dim=1, keepdim=True) / norm

    def compute_domain_logits(self, poses, detach=False):
        if poses is None:
            return None

        if detach:
            with torch.no_grad():
                pred = self.discriminator(poses.to(self.device))
        else:
            pred = self.discriminator(poses.to(self.device))

        return self._weighted_logits(pred)

    def train_discriminator_step(self, real_poses,
                                 fake_poses,
                                 real_domains=None, fake_domains=None):

        self.optimizer.zero_grad()

        batch_size = fake_poses.shape[0]
        real_poses_tensor = self._get_real_batch(batch_size, real_poses)

        fake_poses = fake_poses.to(self.device)

        real_pred = self.discriminator(real_poses_tensor)  # [B, 22]
        fake_pred = self.discriminator(fake_poses.detach())

        if real_domains is not None and fake_domains is not None:
            real_targets = real_domains.to(self.device).float().view(-1, 1)
            fake_targets = fake_domains.to(self.device).float().view(-1, 1)
            real_logits = self._weighted_logits(real_pred)
            fake_logits = self._weighted_logits(fake_pred)
            real_loss = F.binary_cross_entropy_with_logits(real_logits, real_targets)
            fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
        else:
            real_loss = torch.mean((real_pred - 1) ** 2)
            fake_loss = torch.mean(fake_pred ** 2)

        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        self.optimizer.step()

        real_acc = (self._weighted_logits(real_pred).squeeze(1) > 0.5).float().mean()
        fake_acc = (self._weighted_logits(fake_pred).squeeze(1) < 0.5).float().mean()

        return {
            'disc_loss': disc_loss.item(),
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'accuracy': (real_acc + fake_acc) / 2
        }
    
    def compute_amp_reward(self, fake_poses, detach=True, apply_sigmoid=True):
        if detach:
            with torch.no_grad():
                pred = self.discriminator(fake_poses.to(self.device))
        else:
            params = list(self.discriminator.parameters())
            requires_grad = [p.requires_grad for p in params]

            for p in params:
                p.requires_grad_(False)

            try:
                with torch.enable_grad():
                    pred = self.discriminator(fake_poses.to(self.device))
            finally:
                for p, flag in zip(params, requires_grad):
                    p.requires_grad_(flag)

        if apply_sigmoid:
            pred = self._weighted_logits(pred)
            rewards = torch.sigmoid(pred)
            rewards = 2.0 * rewards - 1.0  # [-1, 1]
            return rewards


        return pred
    
    def save_discriminator(self, filepath):
        torch.save({
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Discriminator saved to {filepath}")
    
    def load_discriminator(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Discriminator loaded from {filepath}")

    def load_discriminator(self, checkpoint_path, strict=True, load_optimizer=False):
        if not checkpoint_path:
            raise ValueError('No checkpoint path provided for AMP discriminator.')
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"AMP discriminator checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = None

        if isinstance(checkpoint, dict):
            for key in ['discriminator_state_dict', 'state_dict', 'discriminator']:
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break
            if state_dict is None:
                tensor_items = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
                if tensor_items:
                    state_dict = tensor_items
                else:
                    raise RuntimeError('Checkpoint does not contain discriminator weights.')
        else:
            state_dict = checkpoint

        self.discriminator.load_state_dict(state_dict, strict=strict)

        if load_optimizer and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Loaded AMP discriminator weights from {checkpoint_path}")


class InteractionAMPDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, features):
        return self.net(features)


class InteractionAMPTrainer:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.discriminator = None
        self.optimizer = None
        self.feature_dim = None

    def _ensure_modules(self, feature_dim):
        if feature_dim is None:
            if self.feature_dim is None:
                raise ValueError('feature_dim must be provided before the discriminator is initialised.')
            feature_dim = self.feature_dim

        if self.discriminator is None or self.feature_dim != feature_dim:
            self.feature_dim = feature_dim
            self.discriminator = InteractionAMPDiscriminator(feature_dim).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=1e-4,
                betas=(0.5, 0.999)
            )

    def train_discriminator_step(self, real_features, fake_features):
        if real_features is None or fake_features is None:
            return None
        if real_features.numel() == 0 or fake_features.numel() == 0:
            return None

        real_features = real_features.to(self.device)
        fake_features = fake_features.to(self.device)
        self._ensure_modules(fake_features.shape[-1])

        self.optimizer.zero_grad()

        real_pred = self.discriminator(real_features)
        fake_pred = self.discriminator(fake_features.detach())

        real_loss = torch.mean((real_pred - 1) ** 2)
        fake_loss = torch.mean(fake_pred ** 2)
        disc_loss = real_loss + fake_loss

        disc_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            real_acc = (real_pred.mean(dim=1) > 0.5).float().mean()
            fake_acc = (fake_pred.mean(dim=1) < 0.5).float().mean()

        return {
            'disc_loss': disc_loss.item(),
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'accuracy': (real_acc + fake_acc) / 2
        }

    def compute_reward(self, fake_features, detach=False):
        if fake_features is None:
            return torch.zeros((0, 1), device=self.device)
        if fake_features.numel() == 0:
            return torch.zeros((0, 1), device=fake_features.device)

        fake_features = fake_features.to(self.device)
        self._ensure_modules(fake_features.shape[-1])

        if detach:
            with torch.no_grad():
                pred = self.discriminator(fake_features)
        else:
            pred = self.discriminator(fake_features)

        rewards = torch.sigmoid(pred)
        rewards = 2.0 * rewards - 1.0
        return rewards

    def save_discriminator(self, filepath):
        if self.discriminator is None or self.optimizer is None or self.feature_dim is None:
            raise RuntimeError('Interaction AMP discriminator has not been initialised; cannot save.')

        torch.save({
            'feature_dim': self.feature_dim,
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Interaction AMP discriminator saved to {filepath}")

    def load_discriminator(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        feature_dim = checkpoint.get('feature_dim')
        if feature_dim is None:
            raise KeyError('feature_dim missing from interaction AMP checkpoint.')

        self._ensure_modules(feature_dim)
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Interaction AMP discriminator loaded from {filepath}")