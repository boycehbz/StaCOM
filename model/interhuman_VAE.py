
import torch
from torch import nn
from utils.rotation_conversions import *
from model.vqvae.quantize_cnn import QuantizeEMAReset

from model.utils import *
from model.blocks import *

class interhuman_VAE(nn.Module):
    def __init__(self, smpl, num_joints=21, latentD=256, frame_length=16, n_layers=1, hidden_size=256, bidirectional=True,):
        super(interhuman_VAE, self).__init__()
        self.smpl = smpl

        self.use_mask = False
        if self.use_mask:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, 313))

        num_agent = 2

        self.num_frames = frame_length
        self.latent_dim = 256
        self.ff_size = self.latent_dim * 2
        self.num_layers = 4
        self.num_heads = 8
        self.dropout = 0.1
        self.activation = 'gelu'
        self.input_feats = 157 #144 + 10 + 3

        self.feature_emb_dim = 256

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)

        # Input Embedding
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)

        self.mu_linear = nn.Sequential(nn.Linear(self.latent_dim, latentD),
                                        )
        self.var_linear = nn.Sequential(nn.Linear(self.latent_dim, latentD),
                                        )

        self.motion_decode = nn.Linear(latentD, self.latent_dim)

        self.decoder_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.decoder_blocks.append(TransformerBlock(num_heads=self.num_heads,latent_dim=self.latent_dim, dropout=self.dropout, ff_size=self.ff_size))

        self.encoder_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.encoder_blocks.append(TransformerBlock(num_heads=self.num_heads,latent_dim=self.latent_dim, dropout=self.dropout, ff_size=self.ff_size))

        out_dim = self.input_feats #24 * 6

        self.head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, out_dim),
        )
        # self.cam_head = nn.Sequential(
        #     nn.LayerNorm(self.latent_dim),
        #     nn.Linear(self.latent_dim, 3),
        # )
        # self.shape_head = nn.Sequential(
        #     nn.LayerNorm(self.latent_dim),
        #     nn.Linear(self.latent_dim, 10),
        # )

    def distribution(self, x):
        mean = self.mu_linear(x)
        std = self.var_linear(x)
        q_z = torch.distributions.normal.Normal(mean, F.softplus(std))
        z = q_z.rsample()
        return q_z, z

    def distribution_mean(self, x):
        mean = self.mu_linear(x)
        return mean

    def inference(self, pose, shape, trans):
        batch_size, frame_length, num_people = pose.shape[:3]

        init_trans = trans[:,0,0,:].detach()
        trans = trans - init_trans[:,None,None,:]

        origin_rot = pose[:,0,0,:3]
        origin_rot_matrix = axis_angle_to_matrix(origin_rot.reshape(-1, 3))
        origin_rot_matrix_inv = torch.linalg.inv(origin_rot_matrix)

        for b in range(batch_size):
            for i in range(num_people):
                poses_root_torch = pose[b,:,i,:3]
                all_matrix = axis_angle_to_matrix(poses_root_torch.reshape(-1, 3))

                all_matrix = torch.matmul(origin_rot_matrix_inv[b], all_matrix)
                poses_root_torch = matrix_to_axis_angle(all_matrix).reshape(-1, 3)
                pose[b,:,i,:3] = poses_root_torch

                trans_torch = trans[b,:,i,:]
                trans_torch = trans_torch - trans[b,0,0]
                trans_torch = torch.matmul(origin_rot_matrix_inv[b], trans_torch.reshape(-1, 3).T).T
                trans[b,:,i,:] = trans_torch.reshape(-1, 3)


        # pose = pose.reshape(-1, 72)
        # shape = shape.reshape(-1, 10)
        # trans = trans.reshape(-1, 3)
        # pred_verts, pred_joints = self.smpl(shape, pose, trans, halpe=True)


        # pred = {'pred_pose':pose,\
        #         'pred_shape':shape,\
        #         'pred_cam_t':trans,\
        #         'pred_verts':pred_verts,\
        #         'pred_joints':pred_joints,\
        #          }

        # return pred


        pose_6d = pose.reshape(-1, 3)
        pose_6d = axis_angle_to_matrix(pose_6d)
        pose_6d = matrix_to_rotation_6d(pose_6d)
        pose_6d = pose_6d.reshape(batch_size, frame_length, num_people, -1)

        x = torch.cat([pose_6d, shape, trans], dim=-1)

        B, T = batch_size, frame_length
        x_a, x_b = x[:,:,0], x[:,:,1]

        mask = None
        if mask is not None:
            mask = mask[...,0]

        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)

        if mask is None:
            mask = torch.ones(B, T).to(x_a.device)
        key_padding_mask = ~(mask > 0.5)

        for i,block in enumerate(self.encoder_blocks):
            h_a = block(h_a_prev, h_b_prev, b_emb, key_padding_mask)
            h_b = block(h_b_prev, h_a_prev, a_emb, key_padding_mask)
            h_a_prev = h_a
            h_b_prev = h_b

        z_a = self.distribution_mean(h_a)
        z_b = self.distribution_mean(h_b)

        h_a_prev = self.motion_decode(z_a)
        h_b_prev = self.motion_decode(z_b)

        for i,block in enumerate(self.decoder_blocks):
            h_a = block(h_a_prev, h_b_prev, h_b_prev, key_padding_mask)
            h_b = block(h_b_prev, h_a_prev, h_a_prev, key_padding_mask)
            h_a_prev = h_a
            h_b_prev = h_b

        features = torch.cat([h_a[:,:,None], h_b[:,:,None]], dim=2)
        features = features.reshape(batch_size*frame_length*num_people, -1)

        xc = features

        pred_pose = self.head(xc)
        pred_shape = self.shape_head(xc).view(-1, 10)
        pred_cam = self.cam_head(xc).view(-1, 3)

        pred_rotmat = rotation_6d_to_matrix(pred_pose).view(-1, 24, 3, 3)
        pred_pose =  matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(-1, 72)

        # pred_pose = pose
        # pred_shape = shape
        # pred_cam = trans

        pred_pose = pred_pose.reshape(batch_size, frame_length, num_people, -1)
        pred_shape = pred_shape.reshape(batch_size, frame_length, num_people, -1)
        pred_cam = pred_cam.reshape(batch_size, frame_length, num_people, -1)

        pred_trans = pred_cam

        for b in range(batch_size):
            for i in range(num_people):
                poses_root_torch = pred_pose[b,:,i,:3]
                all_matrix = axis_angle_to_matrix(poses_root_torch.reshape(-1, 3))

                all_matrix = torch.matmul(origin_rot_matrix[b], all_matrix)
                poses_root_torch = matrix_to_axis_angle(all_matrix).reshape(-1, 3)
                pred_pose[b,:,i,:3] = poses_root_torch

                trans_torch = pred_trans[b,:,i,:]
                trans_torch = torch.matmul(origin_rot_matrix[b], trans_torch.reshape(-1, 3).T).T
                trans_torch = trans_torch + pred_trans[b,0,0]
                pred_trans[b,:,i,:] = trans_torch.reshape(-1, 3)


        pred_trans = pred_trans + init_trans[:,None,None,:]

        pred_pose_6d = pred_pose.reshape(-1, 3)
        pred_pose_6d = axis_angle_to_matrix(pred_pose_6d)
        pred_pose_6d = matrix_to_rotation_6d(pred_pose_6d)
        pred_pose_6d = pred_pose_6d.reshape(batch_size, frame_length, num_people, -1)


        # x_t_updated = torch.cat([pred_pose_6d, pred_shape, pred_trans], dim=-1)

        # pred_pose = pred_pose.reshape(-1, 72)
        # pred_shape = pred_shape.reshape(-1, 10)
        # pred_trans = pred_trans.reshape(-1, 3)
        # temp_trans = torch.zeros_like(pred_trans)

        # pred_verts, pred_joints = self.smpl(pred_shape, pred_pose, temp_trans, halpe=True)

        # pred = {'pred_pose':pred_pose,\
        #         'pred_shape':pred_shape,\
        #         'pred_cam_t':pred_trans,\
        #         'pred_verts':pred_verts,\
        #         'pred_joints':pred_joints,\
        #          }

        return pred_pose_6d, pred_shape, pred_trans

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x_masked.shape[1], 1)
        x_masked = torch.cat([x_masked, mask_tokens], dim=1)  # no cls token
        x_masked = torch.gather(x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        return x_masked

    def forward(self, data):

        batch_size, frame_length, agent_num = data['x'].shape[:3]
        num_valid = batch_size * frame_length * agent_num

        x = data['x']

        B, T = batch_size, frame_length
        x_a, x_b = x[:,:,0], x[:,:,1]

        if self.use_mask and self.training:
            x_a = self.random_masking(x_a, 0.5)
            x_b = self.random_masking(x_b, 0.5)

        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)


        key_padding_mask = torch.ones(B, T).to(x_a.device)
        key_padding_mask = ~(key_padding_mask > 0.5)

        for i,block in enumerate(self.encoder_blocks):
            h_a = block(h_a_prev, h_b_prev, b_emb, key_padding_mask)
            h_b = block(h_b_prev, h_a_prev, a_emb, key_padding_mask)
            h_a_prev = h_a
            h_b_prev = h_b

        h_a_prev = self.motion_decode(h_a)
        h_b_prev = self.motion_decode(h_b)

        for i,block in enumerate(self.decoder_blocks):
            h_a = block(h_a_prev, h_b_prev, h_b_prev, key_padding_mask)
            h_b = block(h_b_prev, h_a_prev, h_a_prev, key_padding_mask)
            h_a_prev = h_a
            h_b_prev = h_b

        features = torch.cat([h_a[:,:,None], h_b[:,:,None]], dim=2)
        features = features.reshape(batch_size*frame_length*agent_num, -1)

        xc = features

        xc = self.head(xc).view(batch_size, frame_length, agent_num, -1)
        # pred_shape = self.shape_head(xc).view(batch_size, frame_length, agent_num, -1)
        # pred_cam = self.cam_head(xc).view(batch_size, frame_length, agent_num, -1)

        # incremental
        incremental = False
        if incremental:
            for f in range(frame_length-1):
                pred_pose[:,f+1] = pred_pose[:,f] + pred_pose[:,f+1]
                pred_cam[:,f+1] = pred_cam[:,f] + pred_cam[:,f+1]

            pred_shape = pred_shape.mean(dim=1)
            pred_shape = pred_shape[:,None,:,:].repeat(1,frame_length, 1, 1)

            pred_pose = pred_pose.reshape(-1, 144)
            pred_shape = pred_shape.reshape(-1, 10)
            pred_cam = pred_cam.reshape(-1, 3)

        if not self.training:
            pred_pose = xc[:,:,:,:144]
            pred_shape = xc[:,:,:,144:154]
            pred_trans = xc[:,:,:,154:157]

            pred_shape = pred_shape.reshape(-1, 10)
            pred_trans = pred_trans.reshape(-1, 3)

            pred_rotmat = rotation_6d_to_matrix(pred_pose).view(-1, 24, 3, 3)
            pred_pose =  matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(-1, 72)

            pred_verts, pred_joints = self.smpl(pred_shape, pred_pose, pred_trans, halpe=True)

            pred = {'pred_pose':pred_pose,\
                    'pred_shape':pred_shape,\
                    'pred_cam_t':pred_trans,\
                    'pred_rotmat':pred_rotmat,\
                    'pred_verts':pred_verts,\
                    'pred_joints':pred_joints,\
                    # 'q_a':q_a,\
                    # 'q_b':q_b,\
                    }
            
        else:
            pred = {'pred_x':xc,\
                    # 'q_a':q_a,\
                    # 'q_b':q_b,\
                    }

        return pred

