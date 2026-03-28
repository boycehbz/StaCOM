import torch
from torch import nn
from utils.imutils import cam_crop2full, vis_img
from utils.geometry import perspective_projection
from utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix
from model.utils import *
from model.blocks import *
import smplx
import math
import numpy as np



############### flow matching with BPS as condition ###############
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class InterhumanFlow_BPS(nn.Module):
    def __init__(
        self,
        smpl,
        num_joints=21,
        latentD=32,
        frame_length=16,
        n_layers=1,
        hidden_size=256,
        bidirectional=True,
        use_object_attention=True,
    ):
        super().__init__()
        self.smpl = smpl
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_frame = frame_length
        num_agent = 2
        self.eval_initialized = False
        num_timesteps = 100
        beta_scheduler = 'cosine'
        self.timestep_respacing = 'ddim5'

        self.sampling_num_steps = 10
        self.integration_strength = 1.0 # 0.05

        # Use float64 for accuracy.
        betas = get_named_beta_schedule(beta_scheduler, num_timesteps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.sampler = UniformSampler(num_timesteps)

        self.cfg_weight = 3.5
        self.num_frames = frame_length
        self.latent_dim = 512
        self.ff_size = self.latent_dim * 2
        self.num_layers = 8
        self.num_heads = 8
        self.dropout = 0.1
        self.activation = 'gelu'
        self.input_feats = 144 + 10 + 3
        self.time_embed_dim = 1024
        self.use_object_attention = use_object_attention
        self.feature_emb_dim = 256
        self.bps_feature_dim = 1024
        self.max_agents = 2
        self.object_pose_feature_dim = 12
        self.object_condition_dim = self.object_pose_feature_dim + self.bps_feature_dim

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)

        time_dim = 512
        sinu_pos_emb = SinusoidalPosEmb(256)
        fourier_dim = 256
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input Embedding
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.feature_embed = nn.Linear(self.feature_emb_dim, self.latent_dim)

        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.blocks.append(TransformerBlock(num_heads=self.num_heads,latent_dim=self.latent_dim, dropout=self.dropout, ff_size=self.ff_size))
        # Output Module
        self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))

        img_embed_dim = 12
        out_dim = 24 * 6
        hidden_dim = 256
        self.point_project = nn.Linear(self.latent_dim, hidden_dim)
        
        self.project = nn.Sequential(
            nn.LayerNorm(img_embed_dim),
            nn.Linear(img_embed_dim, hidden_dim),
        )
        self.bps_project = nn.Sequential(
            nn.LayerNorm(self.bps_feature_dim),
            nn.Linear(self.bps_feature_dim, hidden_dim),
        )
        self.object_feature_proj = nn.Sequential(
            nn.LayerNorm(self.object_condition_dim),
            nn.Linear(self.object_condition_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, out_dim),
        )
        self.cam_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim , 3),
        )
        self.shape_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, 10),
        )
        
        self.smplx_model = smplx.create(
        model_path='./data/SMPLX_NEUTRAL.pkl',
        model_type='smplx',
        gender='neutral',
        use_pca=False,
        ext='pkl',
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_transl=True
        ).to(self.device)

    def get_joints_light(self, pred_shape, pred_pose, pred_trans):
        global_orient = pred_pose[:, :3]
        body_pose = pred_pose[:, 3:66]
        
        output = self.smplx_model(
            betas=pred_shape,
            global_orient=global_orient,
            body_pose=body_pose,
            transl=pred_trans,
            return_verts=False 
        )
        return output.joints
    
    def init_eval(self,):
    
        use_timesteps = set(space_timesteps(self.num_timesteps, self.timestep_respacing))
        self.timestep_map = []

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        self.test_betas = np.array(new_betas)

        self.num_timesteps_test = int(self.test_betas.shape[0])

        test_alphas = 1.0 - self.test_betas
        self.test_alphas_cumprod = np.cumprod(test_alphas, axis=0)
        self.test_alphas_cumprod_prev = np.append(1.0, self.test_alphas_cumprod[:-1])
        self.test_alphas_cumprod_next = np.append(self.test_alphas_cumprod[1:], 0.0)
        assert self.test_alphas_cumprod_prev.shape == (self.num_timesteps_test,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.testsqrt_alphas_cumprod = np.sqrt(self.test_alphas_cumprod)
        self.test_sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.test_alphas_cumprod)
        self.test_log_one_minus_alphas_cumprod = np.log(1.0 - self.test_alphas_cumprod)
        self.test_sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.test_alphas_cumprod)
        self.test_sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.test_alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.test_posterior_variance = (
                self.test_betas * (1.0 - self.test_alphas_cumprod_prev) / (1.0 - self.test_alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.test_posterior_log_variance_clipped = np.log(
            np.append(self.test_posterior_variance[1], self.test_posterior_variance[1:])
        )
        self.test_posterior_mean_coef1 = (
                self.test_betas * np.sqrt(self.test_alphas_cumprod_prev) / (1.0 - self.test_alphas_cumprod)
        )
        self.test_posterior_mean_coef2 = (
                (1.0 - self.test_alphas_cumprod_prev)
                * np.sqrt(test_alphas)
                / (1.0 - self.test_alphas_cumprod)
        )


    def condition_process(self, data):
        img_info = {}

        batch_size, frame_length, _ = data['x'].shape[:3]

        device = data['x'].device

        cond_features = []

        if 'obj_pose' in data:
            obj_pose = data['obj_pose']
            obj_rot = obj_pose[..., :3, :3].reshape(batch_size, frame_length, -1)
            obj_trans = obj_pose[..., :3, 3]
            object_pose_feat = torch.cat([obj_rot, obj_trans], dim=-1)
            traj_cond = self.project(object_pose_feat)
        else:
            object_pose_feat = torch.zeros(
                batch_size, frame_length, self.object_pose_feature_dim, device=device
            )
            traj_cond = torch.zeros(
                batch_size, frame_length, self.feature_emb_dim, device=device
            )

        cond_features.append(traj_cond)

        if 'obj_bps' in data:
            bps_cond_raw = data['obj_bps'].reshape(batch_size, frame_length, -1)
            if bps_cond_raw.shape[-1] != self.bps_feature_dim:
                raise ValueError(
                    f"Unexpected BPS feature dimension {bps_cond_raw.shape[-1]} (expected {self.bps_feature_dim})"
                )
        else:
            bps_cond_raw = torch.zeros(
                batch_size, frame_length, self.bps_feature_dim, device=device
            )

        bps_cond = self.bps_project(bps_cond_raw)
        cond_features.append(bps_cond)

        cond = torch.stack(cond_features, dim=0).mean(dim=0)

        if self.use_object_attention:
            object_condition = torch.cat(
                [
                    object_pose_feat,
                    bps_cond_raw,
                ],
                dim=-1,
            )
            obj_features = self.object_feature_proj(object_condition)
        else:
            obj_features = None

        return cond, obj_features, img_info

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def generate_noise(self, init_pose, noise=None):
        if noise is None:
            noise = torch.randn((init_pose.shape[0], init_pose.shape[1], init_pose.shape[2], self.input_feats), device=init_pose.device, dtype=init_pose.dtype)
        
        mean = init_pose

        return noise, mean

    def input_process(self, data, img_info, mean):
        gt_pose = data['pose_6d']
        gt_shape = data['betas']
        gt_trans = data['gt_cam_t']

        batch_size, frame_length, agent_num = gt_pose.shape[:3]

        gt_shape = gt_shape.reshape(batch_size, frame_length, agent_num, -1)
        gt_trans = gt_trans.reshape(batch_size, frame_length, agent_num, -1)

        x_start = torch.cat([gt_pose, gt_shape, gt_trans], dim=-1)

        x_start = x_start - mean

        return x_start

    def inference(self, x_t, t, cond, obj_features, data, mean):
        batch_size, frame_length, agent_num = data['x'].shape[:3]
        num_valid = batch_size * frame_length * agent_num

        mean = mean.reshape(-1, self.input_feats)

        x_a, x_b = x_t[:,:,0], x_t[:,:,1]
        t = t[:,0,0,0]

        mask = None
        if mask is not None:
            mask = mask[...,0]

        cond_embed = self.feature_embed(cond)
        emb = self.time_mlp(t.reshape(-1))[:, None] + cond_embed
        emb = emb.reshape(batch_size, frame_length, -1)

        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)

        if mask is None:
            mask = torch.ones(batch_size, frame_length).to(x_a.device)
        key_padding_mask = ~(mask > 0.5)

        counterpart_mask = torch.ones(batch_size, frame_length, 1).to(x_a.device)

        for i, block in enumerate(self.blocks):
            h_a = block(h_a_prev, h_b_prev * counterpart_mask, emb, key_padding_mask, obj_features=obj_features)
            h_b = block(h_b_prev, h_a_prev * counterpart_mask, emb, key_padding_mask, obj_features=obj_features)
            h_a_prev = h_a
            h_b_prev = h_b
        
        features = torch.cat([h_a[:,:,None], h_b[:,:,None]], dim=2)
        features = features.reshape(batch_size*frame_length*agent_num, -1)

        xc = features

        pred_pose6d = self.head(xc).view(-1, 144) + mean[:,:144]
        pred_shape = self.shape_head(xc).view(-1, 10) + mean[:,144:154]
        pred_cam = self.cam_head(xc).view(-1, 3) + mean[:,-3:]

        pred_rotmat = rotation_6d_to_matrix(pred_pose6d.reshape(-1,6)).view(-1, 24, 3, 3)
        pred_pose =  matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(-1, 72)

        pred_trans = pred_cam

        pred_u_t = torch.cat([pred_pose6d, pred_shape, pred_trans], dim=1)

        pred_u_t = pred_u_t.reshape(batch_size, frame_length, agent_num, -1)

        return pred_u_t

    def forward(self, data):

        batch_size, frame_length, agent_num = data['x'].shape[:3]
        device = data['x'].device

        # cond, img_info = self.condition_process(data)
        cond, obj_features, img_info = self.condition_process(data)

        init_pose = torch.zeros_like(data['x'])
        noise, mean = self.generate_noise(init_pose)

        x_0 = noise
        x_1 = data['x']

        if self.training:
            
            t = torch.rand(batch_size, device=device)
            t = t[:, None, None, None]
            x_t = (1 - t) * x_0 + t * x_1

            u_t = x_1 - x_0

            # pred_u_t = self.inference(x_t, t, cond, img_info, data, mean)
            pred_u_t = self.inference(x_t, t, cond, obj_features, data, mean)
            
            x_recon = x_t + (1 - t) * pred_u_t

            pred_pose6d = x_recon[...,:144].reshape(-1, 144)
            pred_shape = x_recon[...,144:154].reshape(-1, 10)
            pred_trans = x_recon[...,154:].reshape(-1, 3)

            x = torch.cat([pred_pose6d, pred_shape, pred_trans], dim=1)
            pred = {'pred_x':x,\
                    'pred_pose6d':pred_pose6d,\
                    'pred_shape':pred_shape,\
                    'pred_cam_t':pred_trans,\
                    'u_t':u_t,
                    'pred_u_t':pred_u_t,
                    }

        else:

            x_recon = self.sample_with_condition(
                data, cond, obj_features, mean
            )

            pred_pose6d = x_recon[...,:144].reshape(-1, 144)
            pred_shape = x_recon[...,144:154].reshape(-1, 10)
            pred_trans = x_recon[...,154:].reshape(-1, 3)

            x = torch.cat([pred_pose6d, pred_shape, pred_trans], dim=1)

            pred_rotmat = rotation_6d_to_matrix(pred_pose6d).view(-1, 24, 3, 3)
            pred_pose =  matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(-1, 72)

            pred_verts, pred_joints = self.smpl(pred_shape, pred_pose, pred_trans, halpe=True)

            x = torch.cat([pred_pose6d, pred_shape, pred_trans], dim=1)

            pred = {'pred_pose':pred_pose,\
                    'pred_pose6d':pred_pose6d,\
                    'pred_shape':pred_shape,\
                    'pred_cam_t':pred_trans,\
                    'pred_rotmat':pred_rotmat,\
                    'pred_verts':pred_verts,\
                    'pred_joints':pred_joints,\
                    'pred_x':x,\
                    }

        return pred


    def sample(self, data, cond, obj_features, mean):
        """推理采样
        Args:
            data: 输入数据
            num_steps: 积分步数
            integration_strength: 积分强度，控制每步积分的影响程度 (0.0 ~ 1.0)
        """
        batch_size, frame_length, num_agent = data['data_shape']
        device = data['pose'].device
        
        init_pose = torch.zeros_like(data['x'])
        x_t, mean = self.generate_noise(init_pose)

        dt = 1. / self.sampling_num_steps
        for step in range(self.sampling_num_steps):
            t = torch.ones(batch_size, device=device) * step * dt
            t = t[:, None, None, None]

            pred_u_t = self.inference(x_t, t, cond, obj_features, data, mean)

            pred_u_t = pred_u_t * self.integration_strength

            x_t = x_t + pred_u_t * dt

        x_recon = x_t

        return x_recon - mean

    def sample_with_condition(
        self,
        data,
        cond,
        obj_features,
        mean,
        guidance_weight=200,
        use_contact=True,
    ):
        """
            接触约束
        """

        batch_size, frame_length, num_agent = data['data_shape']
        device = data['pose'].device

        num_steps = self.sampling_num_steps
        integration_strength = self.integration_strength

        init_pose = torch.zeros_like(data['x'])
        x_t, mean = self.generate_noise(init_pose)

        dt = 1. / num_steps
        contact_available = use_contact
        for step in range(num_steps):
            current_x = x_t.detach()

            t = torch.ones(batch_size, device=device) * step * dt
            t = t[:, None, None, None]

            pred_u_t = self.inference(current_x, t, cond, obj_features, data, mean)

            contact_loss = None
            x_t_grad = None

            if contact_available:
                with torch.enable_grad():
                    x_t_grad = current_x.detach().requires_grad_(True)

                    pose6d = x_t_grad[..., :144]
                    shape = x_t_grad[..., 144:154]
                    trans = x_t_grad[..., 154:]

                    pose6d_flat = pose6d.reshape(-1, 6)
                    rotmats = rotation_6d_to_matrix(pose6d_flat).view(-1, 24, 3, 3)
                    pose_axis = matrix_to_axis_angle(rotmats.view(-1, 3, 3)).view(-1, 72)

                    shape_flat = shape.reshape(-1, 10)
                    trans_flat = trans.reshape(-1, 3)

                    pred_verts, pred_joints = self.smpl(shape_flat, pose_axis, trans_flat, halpe=True)

                    contact_loss = self._compute_contact_loss(
                        pred_verts,
                        data,
                        pred_joints,
                    )

            if contact_loss is not None and x_t_grad is not None:
                try:
                    print(f"contact_loss: {contact_loss.detach().item():.6f}")
                except (RuntimeError, AttributeError):
                    print(f"contact_loss: {contact_loss}")
                grad = torch.autograd.grad(
                    contact_loss,
                    x_t_grad,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                pred_u_t = pred_u_t - guidance_weight * grad

            pred_u_t = pred_u_t * integration_strength
            x_t = (current_x + pred_u_t * dt).detach()

        return x_t - mean


    def ddim_sample_loop(self, noise, mean, cond, obj_features, img_info, data, eta=0.0):
        if obj_features is None:
            _, obj_features, _ = self.condition_process(data)

        indices = list(range(self.num_timesteps_test))[::-1]

        img = noise
        preds = []
        for i in indices:
            t = torch.tensor([i] * noise.shape[0], device=noise.device)
            pred = self.ddim_sample(img, t, mean, cond, obj_features, data)
            preds.append(pred)

            # construct x_{t-1}
            pred_pose6d = pred['pred_pose6d']
            pred_shape = pred['pred_shape']
            pred_cam = pred['pred_cam_t']

            model_output = torch.cat([pred_pose6d, pred_shape, pred_cam], dim=-1)
            model_output = model_output.reshape(*img.shape)
            model_output = model_output - mean

            model_variance, model_log_variance = (
                    self.test_posterior_variance,
                    self.test_posterior_log_variance_clipped,
                )
            
            model_variance = extract_into_tensor(model_variance, t, img.shape)
            model_log_variance = extract_into_tensor(model_log_variance, t, img.shape)

            pred_xstart = model_output

            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=img, t=t
            )

            assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == img.shape
            )

            # Usually our model outputs epsilon, but we re-derive it
            # in case we used x_start or x_prev prediction.
            eps = self._predict_eps_from_xstart(img, t, pred_xstart)

            alpha_bar = extract_into_tensor(self.test_alphas_cumprod, t, img.shape)
            alpha_bar_prev = extract_into_tensor(self.test_alphas_cumprod_prev, t, img.shape)
            sigma = (
                    eta
                    * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                    * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            )
            # Equation 12.
            noise, _ = self.generate_noise(torch.zeros_like(model_output))
            mean_pred = (
                    pred_xstart * torch.sqrt(alpha_bar_prev)
                    + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
            )
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
            )  # no noise when t == 0
            sample = mean_pred + nonzero_mask * sigma * noise

            img = sample

        return preds[-1]
    

    def ddim_sample(self, x, ts, mean, cond, obj_features, data):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        pred = self.inference(x, new_ts, cond, obj_features, data, mean)
        return pred
    

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                extract_into_tensor(self.test_posterior_mean_coef1, t, x_t.shape) * x_start
                + extract_into_tensor(self.test_posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.test_posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.test_posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                extract_into_tensor(self.test_sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
        ) / extract_into_tensor(self.test_sqrt_recipm1_alphas_cumprod, t, x_t.shape)


    def compute_temporal_weights(self, obj_poses):
        if obj_poses is None:
            return None
            
        B, T = obj_poses.shape[:2]
        device = obj_poses.device

        base_weight = 0.3
        temporal_weights = torch.full((B, T), base_weight, device=device)

        if T > 1:
            obj_positions = obj_poses[:, :, :3, 3]  # [B, T, 3]

            position_diff = torch.norm(
                obj_positions[:, 1:] - obj_positions[:, :-1], dim=-1)  # [B, T-1]

            movement_threshold = 0.01
            movement_mask = position_diff > movement_threshold
            temporal_weights[:, 1:][movement_mask] = 1.0
        
        return temporal_weights

    def _compute_contact_loss(self, pred_verts, data, pred_joints):

        if isinstance(data, dict):
            contact_points = data.get('contact_points')
            contact_valid = data.get('contact_valid')
            data_shape = data.get('data_shape')
        else:
            contact_points = getattr(data, 'contact_points', None)
            contact_valid = getattr(data, 'contact_valid', None)
            data_shape = getattr(data, 'data_shape', None)

            if contact_points is None and hasattr(data, '__getitem__'):
                try:
                    contact_points = data['contact_points']
                except (KeyError, TypeError, IndexError):
                    pass

            if contact_valid is None and hasattr(data, '__getitem__'):
                try:
                    contact_valid = data['contact_valid']
                except (KeyError, TypeError, IndexError):
                    pass

            if data_shape is None and hasattr(data, '__getitem__'):
                try:
                    data_shape = data['data_shape']
                except (KeyError, TypeError, IndexError):
                    pass


        batch_size, frame_length, num_agent = data_shape
        device = pred_verts.device

        joints = pred_joints.view(batch_size, frame_length, num_agent, -1, 3)
        wrist_indices = [10, 9]
        hand_positions = torch.stack(
            [joints[..., wrist_indices[0], :], joints[..., wrist_indices[1], :]],
            dim=-2
        )

        contact_points = contact_points.to(device)
        contact_valid = contact_valid.to(device)

        if contact_points.ndim == 4:
            # Expand batch dimension for single-sample inputs.
            contact_points = contact_points.unsqueeze(0)
            contact_valid = contact_valid.unsqueeze(0)

        contact_points = contact_points.view(batch_size, frame_length, num_agent, 2, 3)
        contact_valid = contact_valid.view(batch_size, frame_length, num_agent, 2)

        valid_mask = contact_valid > 0
        if not torch.any(valid_mask):
            return None

        diff = hand_positions - contact_points
        squared_dist = (diff ** 2).sum(dim=-1)
        weighted_loss = squared_dist * contact_valid
        per_instance_weight = contact_valid.sum(dim=-1)  # [B, T, N]
        positive_instances = per_instance_weight > 0
        if not torch.any(positive_instances):
            return None

        normalized_loss = weighted_loss.sum(dim=-1) / per_instance_weight.clamp_min(1e-6)
        active_loss = normalized_loss * positive_instances.to(normalized_loss.dtype)

        active_count = positive_instances.sum()
        return active_loss.sum() / active_count.to(active_loss.dtype).clamp_min(1.0)
    

class interhuman_flow_BPS(InterhumanFlow_BPS):
    """Backwards-compatible alias preserving the original snake_case naming."""


__all__ = [
    'InterhumanFlow_BPS',
    'interhuman_flow_BPS',
]
