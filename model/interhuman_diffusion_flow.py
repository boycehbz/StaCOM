import torch
from torch import nn
from torch.nn import functional as F
from utils.imutils import cam_crop2full, vis_img
from utils.geometry import perspective_projection
from utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix
from model.utils import *
from model.blocks import *
import smplx
import math
from affordance_constraint_projector import AffordanceConstraintProjector

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

class interhuman_diffusion_flow(nn.Module):
    def __init__(self, smpl, num_joints=21, latentD=32, frame_length=16, n_layers=1, hidden_size=256, bidirectional=True, affordance_data_dir=None):
        super(interhuman_diffusion_flow, self).__init__()
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

        # calculations for diffusion q(x_t | x_{t-1}) and others
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
        self.use_contact_loss = True
        self.use_affordance_constraints = True
        self.affordance_projector = None
        # self.affordance_projector = AffordanceConstraintProjector(affordance_data_dir, self.device)
        # print(f"✓ Initialized dense affordance constraints from {affordance_data_dir}")
        print("✓ Affordance constraints disabled for testing")
        
        self.flow_projection_steps = 3
        self.flow_learning_rate = 0.05 
        self.affordance_constraint_weight = 2.0
        self.temporal_consistency_weight = 0.5
        self.constraint_timestep_threshold = 30
        self.constraint_probability = 0.8     
        self.use_object_condition = True
        self.use_simple_constraints = True
        self.contact_loss_weight = 0.1
        self.affordance_loss_weight = 0.1
        self.feature_emb_dim = 256

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        # self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        
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

        batch_size, frame_length, agent_num = data['x'].shape[:3]

        if self.use_object_condition and 'obj_pose' in data:
            traj_cond = data['obj_pose'][:,:,:3].reshape(batch_size, frame_length, -1)
            traj_cond = self.project(traj_cond)
            cond = traj_cond
        else:
            cond = torch.zeros(batch_size, frame_length, self.feature_emb_dim,
                            device=data['x'].device)
        
        return cond, img_info

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

    def trans2cam(self, trans, img_info):
        
        
        img_h, img_w = img_info['full_img_shape'][:, 0], img_info['full_img_shape'][:, 1]
        cx, cy, b = img_info['center'][:, 0], img_info['center'][:, 1], img_info['scale'] * 200
        w_2, h_2 = img_w / 2., img_h / 2.

        cam_z = (2 * img_info['focal_length']) / (b * trans[:,2] + 1e-9)

        bs = b * cam_z + 1e-9

        cam_x = trans[:,0] - (2 * (cx - w_2) / bs)
        cam_y = trans[:,1] - (2 * (cy - h_2) / bs)

        cam = torch.stack([cam_z, cam_x, cam_y], dim=-1)

        return cam

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

    def inference(self, x_t, t, cond, img_info, data, mean):
        batch_size, frame_length, agent_num = data['x'].shape[:3]
        num_valid = batch_size * frame_length * agent_num

        mean = mean.reshape(-1, self.input_feats)

        x_a, x_b = x_t[:,:,0], x_t[:,:,1]
        t = t[:,0,0,0]

        mask = None
        if mask is not None:
            mask = mask[...,0]

        emb = self.time_mlp(t.reshape(-1))[:,None] + self.feature_embed(cond)
        emb = emb.reshape(batch_size, frame_length, -1)

        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)

        if mask is None:
            mask = torch.ones(batch_size, frame_length).to(x_a.device)
        key_padding_mask = ~(mask > 0.5)

        counterpart_mask = torch.ones(batch_size, frame_length, 1).to(x_a.device)
        # counterpart_mask[data['single_person']>0] = 0.

        for i,block in enumerate(self.blocks):
            h_a = block(h_a_prev, h_b_prev * counterpart_mask, emb, key_padding_mask)
            h_b = block(h_b_prev, h_a_prev * counterpart_mask, emb, key_padding_mask)
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
        
        # if not self.training:
        #     pred_pose = pred_pose6d

        #     pred_shape = pred_shape.reshape(-1, 10)
        #     pred_trans = pred_trans.reshape(-1, 3)

        #     pred_rotmat = rotation_6d_to_matrix(pred_pose).view(-1, 24, 3, 3)
        #     pred_pose =  matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(-1, 72)

        #     pred_verts, pred_joints = self.smpl(pred_shape, pred_pose, pred_trans, halpe=True)

        #     x = torch.cat([pred_pose6d, pred_shape, pred_trans], dim=1)

        #     pred = {'pred_pose':pred_pose,\
        #             'pred_pose6d':pred_pose6d,\
        #             'pred_shape':pred_shape,\
        #             'pred_cam_t':pred_trans,\
        #             'pred_rotmat':pred_rotmat,\
        #             'pred_verts':pred_verts,\
        #             'pred_joints':pred_joints,\
        #             'pred_x':x,\
        #             # 'q_a':q_a,\
        #             # 'q_b':q_b,\
        #             }
            
        # else:
        #     # print("aa")
        #     pred_joints = self.smpl(pred_shape, pred_pose, pred_trans, halpe=False, joints_only=True)
            # pred_joints = self.get_joints_light(pred_shape, pred_pose, pred_trans)
        pred_u_t = torch.cat([pred_pose6d, pred_shape, pred_trans], dim=1)

        pred_u_t = pred_u_t.reshape(batch_size, frame_length, agent_num, -1)

            # pred = {'pred_x':x,\
            #         'pred_pose6d':pred_pose6d,\
            #         'pred_shape':pred_shape,\
            #         'pred_cam_t':pred_trans,\
            #         'pred_joints': pred_joints,
            #         }

        return pred_u_t


    def visualize(self, pose, shape, pred_cam, data, img_info, t_idx, name='images_phys'):
        import cv2
        from utils.renderer_moderngl import Renderer_HOI
        import os
        from utils.FileLoaders import save_pkl
        from utils.module_utils import save_camparam

        # if t_idx not in [0, 5, 10, 15, 20]:
        #     return

        output = os.path.join('test_debug', name)
        os.makedirs(output, exist_ok=True)

        batch_size, frame_length, agent_num = data['features'].shape[:3]

        pred_rotmat = rotation_6d_to_matrix(pose.reshape(-1, 6)).view(-1, 24, 3, 3)
        pose = matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(-1, 72)

        # convert the camera parameters from the crop camera to the full camera
        img_h, img_w = img_info['img_h'], img_info['img_w']
        focal_length = img_info['focal_length']
        center = img_info['center']
        scale = img_info['scale']

        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)
    
        pred_verts, pred_joints = self.smpl(shape, pose, pred_trans, halpe=True)

        shape = shape.reshape(batch_size*frame_length, agent_num, -1).detach().cpu().numpy()
        pose = pose.reshape(batch_size*frame_length, agent_num, -1).detach().cpu().numpy()
        pred_trans = pred_trans.reshape(batch_size*frame_length, agent_num, -1).detach().cpu().numpy()

        pred_verts = pred_verts.reshape(batch_size*frame_length, agent_num, 6890, 3)
        focal_length = focal_length.reshape(batch_size*frame_length, agent_num, -1)[:,0]
        imgs = data['imgname']

        pred_verts = pred_verts.detach().cpu().numpy()
        focal_length = focal_length.detach().cpu().numpy()

        for index, (img, pred_vert, focal) in enumerate(zip(imgs, pred_verts, focal_length)):
            if index > 0:
                break

            name = img[-40:].replace('\\', '_').replace('/', '_')

            # seq, cam, na = img.split('/')[-3:]
            # if seq != 'sidehug37' or cam != 'Camera64' or na != '000055.jpg':
            #     continue

            img = cv2.imread(img)
            img_h, img_w = img.shape[:2]
            renderer = Renderer(focal_length=focal, center=(img_w/2, img_h/2), img_w=img.shape[1], img_h=img.shape[0], faces=self.smpl.faces, same_mesh_color=True)


            pred_smpl = renderer.render_front_view(pred_vert, bg_img_rgb=img.copy())
            pred_smpl_side = renderer.render_side_view(pred_vert)
            pred_smpl = np.concatenate((img, pred_smpl, pred_smpl_side), axis=1)

            img_path = os.path.join(output, 'images')
            os.makedirs(img_path, exist_ok=True)
            render_name = "%s_%02d_timestep%02d_pred_smpl.jpg" % (name, index, t_idx)
            cv2.imwrite(os.path.join(img_path, render_name), pred_smpl)

            renderer.delete()

            data = {}
            data['pose'] = pose[index]
            data['trans'] = pred_trans[index]
            data['betas'] = shape[index]

            intri = np.eye(3)
            intri[0][0] = focal
            intri[1][1] = focal
            intri[0][2] = img_w / 2
            intri[1][2] = img_h / 2
            extri = np.eye(4)
            
            cam_path = os.path.join(output, 'camparams', name)
            os.makedirs(cam_path, exist_ok=True)
            save_camparam(os.path.join(cam_path, 'timestep%02d_camparams.txt' %t_idx), [intri], [extri])

            path = os.path.join(output, 'params', name)
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, 'timestep%02d_0000.pkl' %t_idx)
            save_pkl(path, data)


    def visualize_sampling(self, x_start, ts, data, img_info, mean, noise):


        device, dtype = ts.device, ts.dtype
        indices = list(range(self.num_timesteps))[::-1]

        for t in indices:
            t_idx = t
            t = torch.from_numpy(np.array([t] * x_start.shape[0])).to(device=device, dtype=dtype)

            x_t = self.q_sample(x_start, t, noise=noise)

            x_t = x_t + mean

            pose = x_t[:,:,:,:144].reshape(-1, 144).contiguous()
            shape = x_t[:,:,:,144:154].reshape(-1, 10).contiguous()
            pred_cam = x_t[:,:,:,154:].reshape(-1, 3).contiguous()

            self.visualize(pose, shape, pred_cam, data, img_info, t_idx)

    def forward(self, data):

        batch_size, frame_length, agent_num = data['x'].shape[:3]
        num_valid = batch_size * frame_length * agent_num
        device = data['x'].device

        cond, img_info = self.condition_process(data)

        init_pose = torch.zeros_like(data['x'])
        noise, mean = self.generate_noise(init_pose)

        x_0 = noise
        x_1 = data['x']

        if self.training:
            
            t = torch.rand(batch_size, device=device)
            t = t[:, None, None, None]
            x_t = (1 - t) * x_0 + t * x_1

            u_t = x_1 - x_0

            pred_u_t = self.inference(x_t, t, cond, img_info, data, mean)
            
            x_recon = x_t + (1 - t) * pred_u_t

            # if self.use_affordance_constraints:
                # x_recon = self.apply_physical_constraints(x_recon, data, mean)

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

            x_recon = self.sample(data, cond, img_info, mean)

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


    def sample(self, data, cond, img_info, mean):
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

            pred_u_t = self.inference(x_t, t, cond, img_info, data, mean)

            pred_u_t = pred_u_t * self.integration_strength

            x_t = x_t + pred_u_t * dt

            # x_t = self.apply_physical_constraints(x_t, data, mean)
            if step >= self.sampling_num_steps - 3:
                x_t = self.apply_physical_constraints(x_t, data, mean)
        
        x_recon = x_t
        
        return x_recon - mean
    

    def ddim_sample_loop(self, noise, mean, cond, img_info, data, eta=0.0):
        indices = list(range(self.num_timesteps_test))[::-1]

        img = noise
        preds = []
        for i in indices:
            t = torch.tensor([i] * noise.shape[0], device=noise.device)
            pred = self.ddim_sample(img, t, mean, cond, img_info, data)
            preds.append(pred)

            # construct x_{t-1}
            pred_pose6d = pred['pred_pose6d']
            pred_shape = pred['pred_shape']
            pred_cam = pred['pred_cam_t']

            # Visualize each diffusion step
            viz_denoising = False
            if viz_denoising:
                self.visualize(pred_pose6d, pred_shape, pred_cam, data, img_info, i, name='images_phys')

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

            # visualization
            viz_sampling = False
            if viz_sampling:
                x_t = img + mean

                pose = x_t[:,:,:,:144].reshape(-1, 144).contiguous()
                shape = x_t[:,:,:,144:154].reshape(-1, 10).contiguous()
                pred_cam = x_t[:,:,:,154:].reshape(-1, 3).contiguous()

                self.visualize(pose, shape, pred_cam, data, img_info, i)

        return preds[-1]


    def ddim_sample(self, x, ts, mean, cond, img_info, data):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        pred = self.inference(x, new_ts, cond, img_info, data, mean)

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
    
    def apply_physical_constraints(self, x_t, data, mean):
        if not self.use_affordance_constraints:
            return x_t
        if not hasattr(self, '_constraint_step_counter'):
            self._constraint_step_counter = 0
        self._constraint_step_counter += 1
        if not self._should_apply_constraints():
            return x_t
        
        # try:
        # print(f"Dense affordance flow applied")
        pose6d = x_t[..., :144].reshape(x_t.shape[0], x_t.shape[1], x_t.shape[2], -1)
        shape = x_t[..., 144:154].reshape(x_t.shape[0], x_t.shape[1], x_t.shape[2], -1)
        cam = x_t[..., 154:].reshape(x_t.shape[0], x_t.shape[1], x_t.shape[2], -1)
        joints = self.get_joints_from_pose(pose6d, shape, cam)
        # hand_positions = self.affordance_projector.get_hand_positions_from_joints(joints)
        hand_positions = self._get_hand_positions_from_joints(joints)
        object_poses = data.get('obj_pose')
        object_ids = data.get('obj_ids', ['unknown'])
        
        if object_poses is not None:
            # projected_x_t = self._apply_dense_affordance_flow_projection(
            #     x_t, joints, hand_positions, object_poses, object_ids)
            if self.use_simple_constraints:
                print(f"Dense affordance flow applied")
                projected_x_t = self._apply_simple_distance_constraints(
                    x_t, joints, hand_positions, object_poses, object_ids, data)
            else:
                projected_x_t = self._apply_dense_affordance_flow_projection(
                    x_t, joints, hand_positions, object_poses, object_ids)
            movement = torch.norm(projected_x_t - x_t)
            if movement > 1e-6:
                print(f"Dense affordance flow applied, movement: {movement.item():.6f}")
            
            return projected_x_t
            
        # except Exception as e:
        #     print(f"Physical constraints error: {e}")
        
        return x_t
    
    def debug_rotation_function(self,pose6d_opt):
        print(f"pose6d_opt before: {pose6d_opt.requires_grad}, {pose6d_opt.grad_fn}")
    
        d6_reshaped = pose6d_opt.reshape(-1, 6)
        print(f"after reshape: {d6_reshaped.requires_grad}, {d6_reshaped.grad_fn}")
        
        a1, a2 = d6_reshaped[..., :3], d6_reshaped[..., 3:]
        print(f"after split: a1 {a1.requires_grad}, a2 {a2.requires_grad}")
        
        b1 = F.normalize(a1, dim=-1)
        print(f"after normalize a1: {b1.requires_grad}, {b1.grad_fn}")
        
        dot_product = (b1 * a2).sum(-1, keepdim=True)
        print(f"dot product: {dot_product.requires_grad}, {dot_product.grad_fn}")
        
        b2 = a2 - dot_product * b1
        print(f"b2 before normalize: {b2.requires_grad}, {b2.grad_fn}")
        
        b2_norm = F.normalize(b2, dim=-1)
        print(f"b2 after normalize: {b2_norm.requires_grad}, {b2_norm.grad_fn}")
        
        b3 = torch.cross(b1, b2_norm, dim=-1)
        print(f"b3: {b3.requires_grad}, {b3.grad_fn}")
        
        result = torch.stack((b1, b2_norm, b3), dim=-2)
        print(f"final result: {result.requires_grad}, {result.grad_fn}")
        pytorch3d_result = rotation_6d_to_matrix(pose6d_opt)
        print(f"pytorch3d result: {pytorch3d_result.requires_grad}, {pytorch3d_result.grad_fn}")
        
        return result
    
    def get_joints_from_pose(self, pose6d, shape, cam):
        original_shape = pose6d.shape[:-1]  # [B, T, A]
        rotmat = rotation_6d_to_matrix(pose6d.reshape(-1, 6)).reshape(-1, 24, 3, 3)
        shape_flat = shape.reshape(-1, shape.shape[-1])  # [B*T*A, 10]
        cam_flat = cam.reshape(-1, cam.shape[-1]) if cam is not None else None  # [B*T*A, 3]

        joints = self.get_joints_from_smpl(rotmat, shape_flat, cam_flat)

        if len(joints.shape) == 3:
            batch_size, frame_length, agent_num = original_shape
            joints = joints.reshape(batch_size, frame_length, agent_num, -1, 3)
        
        return joints
    



    def _should_apply_constraints(self):
        if not self.training:
            return True
        return torch.rand(1).item() < self.constraint_probability

    def _apply_dense_affordance_flow_projection(self, x_t, joints, hand_positions, 
                                            object_poses, object_ids):
        batch_size, seq_len, num_agents, feat_dim = x_t.shape
        
        interaction_mask = self.affordance_projector.detect_interaction_phase(
            object_poses, hand_positions, object_ids)
        
        if not interaction_mask.any():
            return x_t
        
        x_optimized = x_t.clone()
        pose6d = x_t[..., :144].clone()
        pose6d_opt = pose6d.clone().requires_grad_(True)
        shape = x_t[..., 144:154]
        cam = x_t[..., 154:]
        
        for step in range(self.flow_projection_steps):
            joints_new = self.get_joints_from_pose(pose6d_opt, shape, cam)
            hand_positions_new = self.affordance_projector.get_hand_positions_from_joints(joints_new)

            target_hand_positions = self.affordance_projector.apply_affordance_constraints(
                hand_positions_new, object_poses, object_ids, interaction_mask)
        
            constraint_loss = self._compute_dense_affordance_loss(
                hand_positions_new, target_hand_positions, interaction_mask)
            
            temporal_loss = self._compute_temporal_consistency_loss(
                pose6d_opt, pose6d, seq_len)

            total_loss = (self.affordance_constraint_weight * constraint_loss + 
                        self.temporal_consistency_weight * temporal_loss)
            
            if total_loss.item() < 1e-7:
                break 

            total_loss.backward(retain_graph=(step < self.flow_projection_steps - 1))
            
            with torch.no_grad():
                pose6d_opt.data -= self.flow_learning_rate * pose6d_opt.grad
                pose6d_opt.grad.zero_()
        
        x_optimized[..., :144] = pose6d_opt.detach()
        
        return x_optimized

    def _compute_dense_affordance_loss(self, current_hand_pos, target_hand_pos, interaction_mask):
        position_diff = current_hand_pos - target_hand_pos  # [B, T, A, 2, 3]
        squared_distances = torch.sum(position_diff ** 2, dim=-1)  # [B, T, A, 2]
        if interaction_mask is not None:
            masked_distances = squared_distances * interaction_mask.float()
            num_interactions = interaction_mask.sum().float() + 1e-8
            loss = masked_distances.sum() / num_interactions
        else:
            loss = squared_distances.mean()
        
        return loss

    def _compute_temporal_consistency_loss(self, pose6d_opt, pose6d_orig, seq_len):
        base_loss = torch.sum(pose6d_opt * 0.0)  
        
        if seq_len <= 1:
            return base_loss 

        pose_diff = pose6d_opt[:, 1:] - pose6d_opt[:, :-1]
        temporal_smoothness_loss = torch.mean(pose_diff ** 2)

        # pose_deviation_loss = torch.mean((pose6d_opt - pose6d_orig) ** 2)
        
        return temporal_smoothness_loss
    

    def get_joints_from_smpl(self, rotmat, shape, cam):
        batch_size = rotmat.shape[0]
        
        # print(f"Input rotmat.requires_grad: {rotmat.requires_grad}")
        
        global_orient_aa = matrix_to_axis_angle(rotmat[:, 0]).reshape(batch_size, 3)
        # print(f"global_orient_aa.requires_grad: {global_orient_aa.requires_grad}")
        
        body_pose_rotmat = rotmat[:, 1:22]
        body_pose_aa = matrix_to_axis_angle(body_pose_rotmat.reshape(-1, 3, 3)).reshape(batch_size, 63)
        # print(f"body_pose_aa.requires_grad: {body_pose_aa.requires_grad}")
        
        shape_input = shape.reshape(batch_size, 10)
        device = global_orient_aa.device
        
        # print(f"SMPLx model training mode: {self.smplx_model.training}")
        
        output = self.smplx_model(
            betas=shape_input,
            global_orient=global_orient_aa,
            body_pose=body_pose_aa,
            jaw_pose=torch.zeros(batch_size, 3, device=device),
            leye_pose=torch.zeros(batch_size, 3, device=device),
            reye_pose=torch.zeros(batch_size, 3, device=device),
            left_hand_pose=torch.zeros(batch_size, 45, device=device),
            right_hand_pose=torch.zeros(batch_size, 45, device=device),
            expression=torch.zeros(batch_size, 10, device=device),
        )
        joints = output.joints
        # print(f"Output joints.requires_grad: {joints.requires_grad}")
        return joints
    
    def _apply_simple_distance_constraints(self, x_t, joints, hand_positions, 
                                 object_poses, object_ids, data):
        with torch.enable_grad():
            batch_size, seq_len, num_agents, feat_dim = x_t.shape
            
            interaction_mask = self._detect_object_movement_interaction(
                object_poses, hand_positions, data)
            
            if not interaction_mask.any():
                return x_t
            
            x_optimized = x_t.clone()
            pose6d_init = x_t[..., :144].clone().detach()
            trans_init = x_t[..., 154:].clone().detach()
            shape = x_t[..., 144:154].clone().detach()
            
            optimize_joint_indices = [16, 17, 20, 21, 22, 23] 
            optimize_param_indices = []
            for joint_idx in optimize_joint_indices:
                optimize_param_indices.extend(range(joint_idx * 6, (joint_idx + 1) * 6))

            pose6d_opt0 = torch.nn.Parameter(pose6d_init[:, :, 0:1, optimize_param_indices].clone())
            pose6d_opt1 = torch.nn.Parameter(pose6d_init[:, :, 1:2, optimize_param_indices].clone()) 
            trans_opt0 = torch.nn.Parameter(trans_init[:, :, 0:1, :].clone())
            trans_opt1 = torch.nn.Parameter(trans_init[:, :, 1:2, :].clone())
            
            optimizer = torch.optim.Adam([pose6d_opt0, pose6d_opt1, trans_opt0, trans_opt1], lr=0.03)
            
            initial_constraint_loss = self._compute_nearest_point_loss(hand_positions, data, interaction_mask)
            print(f"初始约束损失: {initial_constraint_loss.item():.6f}")
            
            for step in range(5):
                optimizer.zero_grad()
                
                pose6d_full = pose6d_init.clone()
                pose6d_full[:, :, 0, optimize_param_indices] = pose6d_opt0.squeeze(2)
                pose6d_full[:, :, 1, optimize_param_indices] = pose6d_opt1.squeeze(2)
                
                trans_full = trans_init.clone()
                trans_full[:, :, 0, :] = trans_opt0.squeeze(2)
                trans_full[:, :, 1, :] = trans_opt1.squeeze(2)
                
                joints_new = self.get_joints_from_pose(pose6d_full, shape, trans_full)
                hand_positions_new = self._get_hand_positions_from_joints(joints_new)

                constraint_loss = self._compute_nearest_point_loss(hand_positions_new, data, interaction_mask)

                reg_loss = (torch.mean((pose6d_opt0 - pose6d_init[:, :, 0:1, optimize_param_indices]) ** 2) + 
                        torch.mean((pose6d_opt1 - pose6d_init[:, :, 1:2, optimize_param_indices]) ** 2) +
                        torch.mean((trans_opt0 - trans_init[:, :, 0:1, :]) ** 2) +
                        torch.mean((trans_opt1 - trans_init[:, :, 1:2, :]) ** 2))
                
                avg_distance = constraint_loss.item()
                if avg_distance < 0.02:
                    constraint_weight = 10.0
                    reg_weight = 0.1
                else:
                    constraint_weight = 2.0
                    reg_weight = 1.0

                total_loss = constraint_weight * constraint_loss + reg_weight * reg_loss

                current_movement = (torch.norm(pose6d_opt0 - pose6d_init[:, :, 0:1, optimize_param_indices]) + 
                       torch.norm(pose6d_opt1 - pose6d_init[:, :, 1:2, optimize_param_indices]) +
                       torch.norm(trans_opt0 - trans_init[:, :, 0:1, :]) +
                       torch.norm(trans_opt1 - trans_init[:, :, 1:2, :]))
    
                if current_movement > 5.0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.8 

                if step % 8 == 0:
                    trans_movement = (torch.norm(trans_opt0 - trans_init[:, :, 0:1, :]) + 
                                    torch.norm(trans_opt1 - trans_init[:, :, 1:2, :]))
                    print(f"Step{step}: trans_movement={trans_movement:.4f}")
                if step % 5 == 0 or step < 3:
                    print(f"Step{step}: constraint_loss={constraint_loss.item():.6f}, "
                        f"reg_loss={reg_loss.item():.6f}, total_loss={total_loss.item():.6f}")
                
                if constraint_loss.item() < 0.0001:
                    break
                
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_([pose6d_opt0, pose6d_opt1, trans_opt0, trans_opt1], max_norm=1.0)
                
                optimizer.step()

            pose6d_final = pose6d_init.clone()
            pose6d_final[:, :, 0, optimize_param_indices] = pose6d_opt0.squeeze(2).detach()
            pose6d_final[:, :, 1, optimize_param_indices] = pose6d_opt1.squeeze(2).detach()

            trans_final = trans_init.clone()
            trans_final[:, :, 0, :] = trans_opt0.squeeze(2).detach()
            trans_final[:, :, 1, :] = trans_opt1.squeeze(2).detach()

            x_optimized[..., :144] = pose6d_final
            x_optimized[..., 154:] = trans_final
            
            final_joints = self.get_joints_from_pose(pose6d_final, shape, trans_final)
            final_hand_positions = self._get_hand_positions_from_joints(final_joints)
            final_constraint_loss = self._compute_nearest_point_loss(final_hand_positions, data, interaction_mask)
            
            improvement = initial_constraint_loss.item() - final_constraint_loss.item()
            print(f"优化完成 - 初始损失: {initial_constraint_loss.item():.6f}, "
                f"最终损失: {final_constraint_loss.item():.6f}, "
                f"改善: {improvement:.6f}")
            
            return x_optimized
        
    
    def _detect_object_movement_interaction(self, object_poses, hand_positions, data):
        batch_size, seq_len, num_agents = hand_positions.shape[:3]
        interaction_mask = torch.zeros(batch_size, seq_len, num_agents, 2,
                                    dtype=torch.bool, device=hand_positions.device)
        
        for b in range(batch_size):
            for t in range(seq_len):
                object_moving = False
                if t > 0:
                    current_pos = object_poses[b, t, :3, 3]  # [3]
                    prev_pos = object_poses[b, t-1, :3, 3]   # [3] 
                    velocity = torch.norm(current_pos - prev_pos)
                    if velocity > 0.01: 
                        object_moving = True
                
                if object_moving:
                    for agent in range(num_agents):
                        for hand in range(2):
                            interaction_mask[b, t, agent, hand] = True
        
        return interaction_mask
    
    def _compute_nearest_point_loss(self, hand_positions, data, interaction_mask):
        total_loss = hand_positions.sum() * 0.0
        count = 0
        
        for b in range(hand_positions.shape[0]):
            for t in range(hand_positions.shape[1]):
                for agent in range(hand_positions.shape[2]):
                    for hand in range(hand_positions.shape[3]):
                        if interaction_mask[b, t, agent, hand]:
                            hand_pos = hand_positions[b, t, agent, hand]
                            
                            if 'obj_points' in data and 'obj_pose' in data:
                                obj_points_local = data['obj_points'][b, t, :, :3].detach()
                                obj_pose = data['obj_pose'][b, t]
                                ones = torch.ones(obj_points_local.shape[0], 1, device=obj_points_local.device)
                                obj_points_homo = torch.cat([obj_points_local, ones], dim=1)
                                obj_points_world = torch.matmul(obj_points_homo, obj_pose.T)[:, :3]
                                
                                distances = torch.norm(obj_points_world - hand_pos.unsqueeze(0), dim=-1)
                                min_distance = torch.min(distances)
                                if min_distance > 0.05:
                                    contact_loss = min_distance ** 2
                                elif min_distance > 0.001: 
                                    contact_loss = min_distance * 10
                                else:  
                                    contact_loss = 0.001 / (min_distance + 1e-6)

                                total_loss = total_loss + contact_loss
                                count += 1

        print(f"总约束数量: {count}, 平均损失: {(total_loss/max(count,1)).item():.6f}")
        return total_loss / max(count, 1)

    
    def debug_coordinate_systems(self, data):
        batch_idx, time_idx = 0, 0
        
        if 'obj_points' in data and 'obj_pose' in data:
            obj_points_local = data['obj_points'][batch_idx, time_idx, :10, :3]
            print(f"局部坐标范围: x({obj_points_local[:, 0].min():.3f}~{obj_points_local[:, 0].max():.3f})")
            print(f"           y({obj_points_local[:, 1].min():.3f}~{obj_points_local[:, 1].max():.3f})")
            print(f"           z({obj_points_local[:, 2].min():.3f}~{obj_points_local[:, 2].max():.3f})")
            
            obj_pose = data['obj_pose'][batch_idx, time_idx]
            ones = torch.ones(obj_points_local.shape[0], 1, device=obj_points_local.device)
            obj_points_homo = torch.cat([obj_points_local, ones], dim=1)
            obj_points_world = torch.matmul(obj_points_homo, obj_pose.T)[:, :3]
            
            print(f"世界坐标范围: x({obj_points_world[:, 0].min():.3f}~{obj_points_world[:, 0].max():.3f})")
            print(f"           y({obj_points_world[:, 1].min():.3f}~{obj_points_world[:, 1].max():.3f})")
            print(f"           z({obj_points_world[:, 2].min():.3f}~{obj_points_world[:, 2].max():.3f})")
            
            x_t = data['x']
            pose6d = x_t[..., :144]
            shape = x_t[..., 144:154] 
            cam = x_t[..., 154:]
            joints = self.get_joints_from_pose(pose6d, shape, cam)
            hand_positions = self._get_hand_positions_from_joints(joints)
            
            hand_pos = hand_positions[batch_idx, time_idx, 0, 0]
            print(f"手部位置: x({hand_pos[0]:.3f}), y({hand_pos[1]:.3f}), z({hand_pos[2]:.3f})")
        
            dist_local = torch.min(torch.norm(obj_points_local - hand_pos.unsqueeze(0), dim=-1))
            dist_world = torch.min(torch.norm(obj_points_world - hand_pos.unsqueeze(0), dim=-1))
            print(f"使用局部坐标的距离: {dist_local:.4f}m")
            print(f"使用世界坐标的距离: {dist_world:.4f}m")
    
    def _get_hand_positions_from_joints(self, joints):
        batch_size, num_frames, num_agents = joints.shape[:3]
        
        left_wrists = joints[:, :, :, 20, :]
        right_wrists = joints[:, :, :, 21, :] 

        hand_positions = torch.stack([left_wrists, right_wrists], dim=3)
        
        return hand_positions
        
    def manual_normalize(self, x, dim=-1, eps=1e-8):
        # print(f"manual_normalize input: {x.requires_grad}")
        # squared = x * x
        # print(f"squared: {squared.requires_grad}")
        # print(f"manual_normalize input: {x.requires_grad}")
        # print(f"input grad_fn: {x.grad_fn}")
        # print(f"input device: {x.device}")
        # print(f"input dtype: {x.dtype}")
        # print(f"input is_leaf: {x.is_leaf}")
        
        test = x + 0
        # print(f"x + 0: {test.requires_grad}")
        
        squared = x * x
        # print(f"squared: {squared.requires_grad}")
        # print(f"squared grad_fn: {squared.grad_fn}")
        
        sum_squared = torch.sum(squared, dim=dim, keepdim=True)
        # print(f"sum_squared: {sum_squared.requires_grad}")
        
        norm = torch.sqrt(sum_squared)
        # print(f"manual norm: {norm.requires_grad}")
        
        result = x / (norm + eps)
        # print(f"manual_normalize output: {result.requires_grad}")
        return result

    def rotation_6d_to_matrix_grad(self, d6):
        # print(f"rotation_6d_to_matrix_grad input d6.requires_grad: {d6.requires_grad}")
        
        d6 = d6.view(-1, 6)
        # print(f"after view: {d6.requires_grad}")
        
        a1, a2 = d6[..., :3], d6[..., 3:]
        # print(f"after split: a1 {a1.requires_grad}, a2 {a2.requires_grad}")
        
        b1 = self.manual_normalize(a1, dim=-1)
        # print(f"after manual_normalize a1: {b1.requires_grad}")
        
        dot_product = (b1 * a2).sum(-1, keepdim=True)
        # print(f"dot product: {dot_product.requires_grad}")
        
        b2 = a2 - dot_product * b1
        # print(f"b2 before normalize: {b2.requires_grad}")
        
        b2 = self.manual_normalize(b2, dim=-1)
        # print(f"b2 after normalize: {b2.requires_grad}")
        
        b3 = torch.cross(b1, b2, dim=-1)
        # print(f"b3: {b3.requires_grad}")
        
        result = torch.stack((b1, b2, b3), dim=-2)
        # print(f"final result: {result.requires_grad}")
        
        return result