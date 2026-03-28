import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.diffusion import InterGenDiffusion
from model.mlp import ObjectDenoiser


class HandObjectContactLoss(nn.Module):
    def __init__(self, 
                 contact_threshold=0.05,
                 movement_threshold=0.01,
                 weight=1.0,
                 smplx_model_path='smplx_n/models/',
                 device='cuda:0'):
        super().__init__()
        self.contact_threshold = contact_threshold
        self.movement_threshold = movement_threshold
        self.weight = weight
        self.device = device
        
        self.init_smplx_model(smplx_model_path)
        self.hand_joint_indices = {
            'left_wrist': 20, 
            'right_wrist': 21, 
        }
    
    def init_smplx_model(self, model_path):
        try:
            import smplx
            self.smplx_model = smplx.create(
                model_path,
                model_type='smplx',
                gender='neutral',
                use_face_contour=False,
                use_pca=False,
                create_global_orient=True,
                create_body_pose=True,
                create_betas=True,
                create_left_hand_pose=True,
                create_right_hand_pose=True,
                create_expression=True,
                create_jaw_pose=True,
                create_leye_pose=True,
                create_reye_pose=True,
                create_transl=True,
                ext='pkl'
            ).to(self.device)
            print("SMPLX model loaded for contact loss")
        except Exception as e:
            print(f"Failed to load SMPLX model: {e}")
            self.smplx_model = None
    
    def get_object_vertices(self, obj_poses, obj_mesh_vertices):
        B, T = obj_poses.shape[:2]
        N = obj_mesh_vertices.shape[0]
        vertices = obj_mesh_vertices.unsqueeze(0).unsqueeze(0).expand(B, T, N, 3)  # [B, T, N, 3]
        vertices_homo = torch.cat([vertices, torch.ones(B, T, N, 1, device=vertices.device)], dim=-1)  # [B, T, N, 4]
        obj_vertices = torch.matmul(vertices_homo.unsqueeze(-2), obj_poses.unsqueeze(-3).transpose(-2, -1))  # [B, T, N, 1, 4]
        obj_vertices = obj_vertices.squeeze(-2)[..., :3]  # [B, T, N, 3]
        return obj_vertices
    
    def detect_object_movement(self, obj_vertices):
        """
        检测物体是否在移动
        """
        if obj_vertices.shape[1] <= 1:
            return torch.zeros(obj_vertices.shape[0], obj_vertices.shape[1], 
                             dtype=torch.bool, device=obj_vertices.device)
        
        obj_centers = torch.mean(obj_vertices, dim=2)  # [B, T, 3]
        center_displacement = obj_centers[:, 1:] - obj_centers[:, :-1]  # [B, T-1, 3]
        movement_magnitude = torch.norm(center_displacement, dim=-1)  # [B, T-1]
        movement_mask = torch.zeros(obj_vertices.shape[0], obj_vertices.shape[1], 
                                  dtype=torch.bool, device=obj_vertices.device)
        movement_mask[:, 1:] = movement_magnitude > self.movement_threshold
        
        return movement_mask
    
    def get_hand_joint_positions(self, global_orient, body_pose, transl):
        B, T = global_orient.shape[:2]
        # print(f"DEBUG: Input shapes - global_orient: {global_orient.shape}, body_pose: {body_pose.shape}, transl: {transl.shape}")
        global_orient_flat = global_orient.reshape(B * T, 3)
        body_pose_flat = body_pose.reshape(B * T, 63)
        transl_flat = transl.reshape(B * T, 3)
        # print(f"DEBUG: Flattened shapes - global_orient_flat: {global_orient_flat.shape}, body_pose_flat: {body_pose_flat.shape}, transl_flat: {transl_flat.shape}")
        betas = torch.zeros(B * T, 10, device=global_orient.device)
        left_hand_pose = torch.zeros(B * T, 45, device=global_orient.device)
        right_hand_pose = torch.zeros(B * T, 45, device=global_orient.device)
        expression = torch.zeros(B * T, 10, device=global_orient.device)
        jaw_pose = torch.zeros(B * T, 3, device=global_orient.device)
        leye_pose = torch.zeros(B * T, 3, device=global_orient.device)
        reye_pose = torch.zeros(B * T, 3, device=global_orient.device)
        # print(f"DEBUG: Default params shapes - betas: {betas.shape}, left_hand_pose: {left_hand_pose.shape}, right_hand_pose: {right_hand_pose.shape}")
        try:
            output = self.smplx_model(
                global_orient=global_orient_flat,
                body_pose=body_pose_flat,
                transl=transl_flat,
                betas=betas,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                expression=expression,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
                return_verts=False
            )
            
            joints = output.joints  # [B*T, num_joints, 3]
            joints = joints.reshape(B, T, -1, 3)  # [B, T, num_joints, 3]
            
            hand_positions = {}
            for name, idx in self.hand_joint_indices.items():
                if idx < joints.shape[2]:
                    hand_positions[name] = joints[:, :, idx, :]  # [B, T, 3]
                else:
                    hand_positions[name] = torch.zeros(B, T, 3, device=joints.device)
            
            return hand_positions
            
        except Exception as e:
            print(f"SMPLX forward pass failed: {e}")
            return {
                'left_wrist': torch.zeros(B, T, 3, device=global_orient.device),
                'right_wrist': torch.zeros(B, T, 3, device=global_orient.device)
            }
    
    def compute_hand_object_distances(self, hand_positions, obj_vertices):
        min_distances = {}
        
        for hand_name, hand_pos in hand_positions.items():
            hand_expanded = hand_pos.unsqueeze(2)  # [B, T, 1, 3]
            distances = torch.norm(hand_expanded - obj_vertices, dim=-1)  # [B, T, N]
            min_dist = torch.min(distances, dim=-1)[0]  # [B, T]
            min_distances[hand_name] = min_dist
        
        return min_distances
    
    def forward(self, p1_motion, p2_motion, obj_poses, obj_mesh_vertices):
        obj_vertices = self.get_object_vertices(obj_poses, obj_mesh_vertices)  # [B, T, N, 3]
        movement_mask = self.detect_object_movement(obj_vertices)  # [B, T]
        
        if not movement_mask.any():
            return torch.tensor(0.0, device=obj_poses.device, requires_grad=True)
        
        total_loss = 0.0
        
        for person_idx, motion in enumerate([p1_motion, p2_motion]):
            global_orient = motion[:, :, :3]        # [B, T, 3]
            body_pose = motion[:, :, 3:66]          # [B, T, 63]
            transl = motion[:, :, 66:69]            # [B, T, 3]
        
            hand_positions = self.get_hand_joint_positions(global_orient, body_pose, transl)
            
            hand_distances = self.compute_hand_object_distances(hand_positions, obj_vertices)
            
            for hand_name, distances in hand_distances.items():
                masked_distances = distances * movement_mask.float()
                contact_violation = F.relu(masked_distances - self.contact_threshold)
                if movement_mask.sum() > 0:
                    hand_loss = contact_violation.sum() / movement_mask.sum().float()
                    total_loss += hand_loss
        
        return self.weight * total_loss


class intergen_baseline(nn.Module):
    def __init__(self, smpl, num_joints=26, frame_length=16):
        super().__init__()
           
        # Fixed sequence length
        self.seq_len = 280
        
        # Motion dimensions
        self.full_motion_dim = 157*2
        self.obj_traj_dim = 12
        self.obj_shape_dim = 44
        
        # Model architecture parameters
        self.latent_dim = 512
        self.num_layers = 8
        self.num_heads = 8
        self.ff_size = 1024
        self.dropout = 0.1
        
        # Diffusion parameters
        self.cfg_weight = 2.0
        self.num_timesteps = 1000
        self.beta_scheduler = 'linear'
        self.sampling_strategy = 'ddim50'
        
        # Contact loss parameters
        self.use_contact_loss = False
        self.contact_loss_weight = 0.1
        
        # Create InterGen-style denoising network
        self.denoise_net = ObjectDenoiser(
            motion_dim=self.full_motion_dim,
            obj_traj_dim=self.obj_traj_dim,
            obj_shape_dim=self.obj_shape_dim,
            latent_dim=self.latent_dim,
            num_frames=self.seq_len,
            ff_size=self.ff_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # Create InterGen-style diffusion model
        self.diffusion = InterGenDiffusion(
            denoise_net=self.denoise_net,
            cfg_weight=self.cfg_weight,
            num_timesteps=self.num_timesteps,
            beta_scheduler=self.beta_scheduler,
            sampling_strategy=self.sampling_strategy
        )
        
        # Initialize Hand-Object Contact Loss
        if self.use_contact_loss:
            device = 'cuda:0'
            # smplx_path = getattr(config.model, 'smplx_model_path', 'smplx_n/models/')
            # self.contact_loss_fn = HandObjectContactLoss(
            #     contact_threshold=getattr(config.model, 'contact_threshold', 0.05),
            #     movement_threshold=getattr(config.model, 'movement_threshold', 0.01),
            #     weight=self.contact_loss_weight,
            #     smplx_model_path=smplx_path,
            #     device=device
            # )
        
        # Cache for loss info
        self._last_loss_info = {}
        
        # Cache for original motion data (used in inference)
        self._cached_original_data = None

    def object_pose_to_features(self, obj_poses):
        # Extract position (translation)
        position = obj_poses[..., :3, 3]  # [B, T, 3]
        
        # Extract rotation matrix (3x3) and flatten
        rotation = obj_poses[..., :3, :3]  # [B, T, 3, 3]
        rotation_flat = rotation.reshape(rotation.shape[0], rotation.shape[1], 9)  # [B, T, 9]
        
        # Concatenate position and rotation
        obj_features = torch.cat([position, rotation_flat], dim=-1)  # [B, T, 12]
        
        return obj_features

    def pad_to_fixed_length(self, data, target_length=280):
        current_length = data.shape[1]
        if current_length >= target_length:
            return data[:, :target_length]
        else:
            # Repeat last frame
            last_frame = data[:, -1:].expand(-1, target_length - current_length, *data.shape[2:])
            return torch.cat([data, last_frame], dim=1)

    def extract_body_poses_from_motions(self, motions):
        # Person 1: global_orient[3] + body_pose[63] + transl[3] = 69
        # Person 2: global_orient[3] + body_pose[63] + transl[3] = 69
        p1_global_orient = motions[:, :, :3]      # [B, T, 3]
        p1_body_pose = motions[:, :, 3:66]        # [B, T, 63]
        p1_transl = motions[:, :, 66:69]          # [B, T, 3]
        
        p2_global_orient = motions[:, :, 69:72]   # [B, T, 3]
        p2_body_pose = motions[:, :, 72:135]      # [B, T, 63]
        p2_transl = motions[:, :, 135:138]        # [B, T, 3]
        
        body_poses = torch.cat([p1_body_pose, p2_body_pose], dim=-1)  # [B, T, 126]
        global_orients = torch.cat([p1_global_orient, p2_global_orient], dim=-1)  # [B, T, 6]
        transls = torch.cat([p1_transl, p2_transl], dim=-1)  # [B, T, 6]
        
        return body_poses, global_orients, transls

    def combine_poses_with_original_motion(self, generated_body_poses, original_global_orients, original_transls):
        p1_body_pose = generated_body_poses[:, :, :63]    # [B, T, 63]
        p2_body_pose = generated_body_poses[:, :, 63:]    # [B, T, 63]
        
        p1_global_orient = original_global_orients[:, :, :3]   # [B, T, 3]
        p2_global_orient = original_global_orients[:, :, 3:]   # [B, T, 3]
        
        p1_transl = original_transls[:, :, :3]    # [B, T, 3]
        p2_transl = original_transls[:, :, 3:]    # [B, T, 3]

        p1_motion = torch.cat([p1_global_orient, p1_body_pose, p1_transl], dim=-1)  # [B, T, 69]
        p2_motion = torch.cat([p2_global_orient, p2_body_pose, p2_transl], dim=-1)  # [B, T, 69]
        
        full_motions = torch.cat([p1_motion, p2_motion], dim=-1)  # [B, T, 138]
        
        return full_motions, p1_motion, p2_motion

    def prepare_data(self, batch):
        """
        Prepare data for training/inference
        """
        # Get motion data and pad to fixed length
        motions = batch["motions"]  # [B, T, 138]
        motions = self.pad_to_fixed_length(motions, self.seq_len)  # [B, 280, 138]
        body_poses, global_orients, transls = self.extract_body_poses_from_motions(motions)
        full_motions = motions
        # Get object trajectory and pad to fixed length
        obj_poses = batch["obj_poses"]  # [B, T, 4, 4]
        obj_poses = self.pad_to_fixed_length(obj_poses, self.seq_len)  # [B, 280, 4, 4]
        obj_trajectory = self.object_pose_to_features(obj_poses)  # [B, 280, 12]
        
        # Object shape features
        obj_shape = batch.get("object_features", None)  # [B, shape_dim]
        if obj_shape is None:
            B = motions.shape[0]
            obj_shape = torch.zeros(B, self.obj_shape_dim, device=motions.device)
        
        # Cache original data for inference
        self._cached_original_data = {
            'global_orients': global_orients,
            'transls': transls,
            'full_motions': motions,
            'obj_poses': obj_poses
        }
        
        return full_motions, obj_trajectory, obj_shape

    def forward(self, data):
        """
        Forward pass
        """
        batch_size, frame_length, agent_num = data['x'].shape[:3]
        full_motions = data['x'].reshape(batch_size, frame_length, -1)
        obj_trajectory = data['obj_pose'][:,:,:3].reshape(batch_size, frame_length, -1)
        obj_shape = None
        
        if self.training:
            pred = self._training_forward(full_motions, obj_trajectory, obj_shape, data)
        else:
            pred = self._inference_forward(obj_trajectory, obj_shape)

        return pred

    def _training_forward(self, full_motions, obj_trajectory, obj_shape, batch):
        diffusion_loss = self.diffusion.training_loss(
            x_start=full_motions,  # [B, T, 138]
            obj_trajectory=obj_trajectory,
            obj_shape=obj_shape
        )
        total_loss = diffusion_loss
        affordance_loss = self._compute_affordance_loss(batch)
        total_loss = total_loss + affordance_loss
        self._last_loss_info = {
            'diffusion_loss': diffusion_loss.item(),
            # 'contact_loss': 0.0,
            'total_loss': total_loss.item(),
            'contact_loss_weight': getattr(self, 'contact_loss_weight', 0.0)
        }
        return total_loss

    def _inference_forward(self, obj_trajectory, obj_shape):
        B = obj_trajectory.shape[0]
        generated_full_motions = self.diffusion.sample(
            shape=(B, self.seq_len, self.full_motion_dim),  # [B, 280, 138]
            obj_trajectory=obj_trajectory,
            obj_shape=obj_shape
        )
        
        p1_motion = generated_full_motions[:, :, :69]    # [B, 280, 69]
        p2_motion = generated_full_motions[:, :, 69:]    # [B, 280, 69]
        
        return {
            "motions": generated_full_motions,
            "p1_motion": p1_motion,
            "p2_motion": p2_motion,
            "note": "Complete motion generation"
        }

    def generate(self, obj_trajectory, obj_shape, num_samples=1):
        """
        Generate motion sequences
        """
        self.eval()
        
        with torch.no_grad():
            obj_trajectory = self.pad_to_fixed_length(obj_trajectory, self.seq_len)
            
            all_samples = []
            
            for _ in range(num_samples):
                result = self._inference_forward(obj_trajectory, obj_shape)
                all_samples.append(result)
            
            return all_samples

    def get_loss_dict(self):
        return self._last_loss_info.copy()

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            # "body_pose_dim": self.body_pose_dim,
            "full_motion_dim": self.full_motion_dim,
            "obj_traj_dim": self.obj_traj_dim,
            "obj_shape_dim": self.obj_shape_dim,
            "seq_len": self.seq_len,
            "latent_dim": self.latent_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ff_size": self.ff_size,
            "cfg_weight": self.cfg_weight,
            "num_timesteps": self.num_timesteps,
            "use_contact_loss": self.use_contact_loss,
            "contact_loss_weight": self.contact_loss_weight,
            "model_type": "intergen_style_motion_generator_with_contact_loss",
            "note": "InterGen-style architecture with Hand-Object Contact Loss"
        }


# # Backward compatibility
# SimpleMotionGenerator = InterGenStyleMotionGenerator
# ObjectConditionedMotionGenerator = InterGenStyleMotionGenerator
# ConditionedMotionGenerator = InterGenStyleMotionGenerator