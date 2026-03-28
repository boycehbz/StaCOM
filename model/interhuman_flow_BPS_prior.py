from torch import nn
from utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix, axis_angle_to_matrix, matrix_to_rotation_6d
from model.utils import *
from model.blocks import *
import math
import numpy as np
import os
import sys
from tqdm.auto import tqdm

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

class InterhumanFlow_BPS_DualDisc(nn.Module):
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
        self.eval_initialized = False
        num_timesteps = 100
        beta_scheduler = 'cosine'
        self.timestep_respacing = 'ddim5'

        self.sampling_num_steps = 30
        self.integration_strength = 1.0

        self.use_cmaes_physics = True
        self.cmaes_sigma = 1.4  
        self.cmaes_times = 20 
        self.cmaes_sample_rollouts = 24
        self._cmaes_bundle = None
        self._smpl2humanoid = [1, 4, 7, 2, 5, 8, 3, 6, 9, 12, 15, 13, 16, 18, 14, 17, 19]

        global_mean = torch.from_numpy(
            np.load('./data/global_mean_235.npy')
        ).float()
        global_std = torch.from_numpy(
            np.load('./data/global_std_235.npy')
        ).float()
        self.register_buffer('global_mean', global_mean)
        self.register_buffer('global_std',  global_std)

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
        self.pose_dim = 144
        self.shape_dim = 10
        self.trans_dim = 3
        self.num_joints = 26
        self.joint_dim = self.num_joints * 3
        self.input_feats = self.pose_dim + self.shape_dim + self.trans_dim + self.joint_dim

        self.time_embed_dim = 1024
        self.use_object_attention = use_object_attention
        self.single_amp_weight = 1.0
        self.feature_emb_dim = 256
        self.bps_feature_dim = 1024
        self.max_agents = 2
        self.num_hand_tokens = 2
        self.object_pose_feature_dim = 12
        self.contact_point_dim = self.max_agents * self.num_hand_tokens * 3
        self.contact_valid_dim = self.max_agents * self.num_hand_tokens
        self.contact_normal_dim = self.max_agents * self.num_hand_tokens * 3
        self.contact_condition_dim = self.contact_point_dim + self.contact_valid_dim
        self.object_condition_dim = (
            self.object_pose_feature_dim
            + self.bps_feature_dim
            + self.contact_point_dim
            + self.contact_normal_dim
            + self.contact_valid_dim
        )

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

        self.motion_embed = nn.Linear(self.input_feats + 3, self.latent_dim)
        self.cond_fusion = nn.Linear(self.feature_emb_dim * 3, self.feature_emb_dim)
        self.feature_embed = nn.Linear(self.feature_emb_dim, self.latent_dim)

        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.blocks.append(TransformerBlock(num_heads=self.num_heads, latent_dim=self.latent_dim, dropout=self.dropout, ff_size=self.ff_size))

        self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))

        img_embed_dim = 12
        out_dim = 24 * 6
        hidden_dim = 256

        self.project = nn.Sequential(
            nn.LayerNorm(img_embed_dim),
            nn.Linear(img_embed_dim, hidden_dim),
        )
        self.bps_project = nn.Sequential(
            nn.LayerNorm(self.bps_feature_dim),
            nn.Linear(self.bps_feature_dim, hidden_dim),
        )
        self.contact_project = nn.Sequential(
            nn.LayerNorm(self.contact_condition_dim),
            nn.Linear(self.contact_condition_dim, hidden_dim),
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
            nn.Linear(self.latent_dim, 3),
        )
        self.shape_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, 10),
        )
        self.joints_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.joint_dim),
        )

    def init_eval(self):
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

        self.testsqrt_alphas_cumprod = np.sqrt(self.test_alphas_cumprod)
        self.test_sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.test_alphas_cumprod)
        self.test_log_one_minus_alphas_cumprod = np.log(1.0 - self.test_alphas_cumprod)
        self.test_sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.test_alphas_cumprod)
        self.test_sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.test_alphas_cumprod - 1)

        self.test_posterior_variance = (
            self.test_betas * (1.0 - self.test_alphas_cumprod_prev) / (1.0 - self.test_alphas_cumprod)
        )
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

    def _get_cmaes_bundle(self, object_mesh_path, object_pose_path):
        if self._cmaes_bundle is not None:
            bundle_mesh_path = self._cmaes_bundle['object_mesh_path']
            bundle_pose_path = self._cmaes_bundle['object_pose_path']
            if bundle_mesh_path == object_mesh_path and bundle_pose_path == object_pose_path:
                return self._cmaes_bundle

        cmaes_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cmaes'))
        if cmaes_root not in sys.path:
            sys.path.insert(0, cmaes_root)

        import gym
        import vclrl_envs
        from utils.smpl_torch_batch import SMPLModel
        from utils.dist_adaptation import CMAES

        model_smpl = SMPLModel(
            device=torch.device('cpu'),
            model_path='./data/SMPL_NEUTRAL.pkl',
            data_type=torch.float32,
        )
        env = gym.make(
            'AMASSSampleEnv-v0',
            render=False,
            data='',
            smpl=model_smpl,
            vis_motion=False,
            object_mesh_path=object_mesh_path,
            object_pose_path=object_pose_path,
            object_mass=5.0,
            object_stability_seconds=0.05,
            object_stability_weight=10.0,
        )

        cmaes_solver = CMAES(env, times=self.cmaes_times, device=torch.device('cpu'), window_size=1)
        self._cmaes_bundle = {
            'env': env,
            'solver': cmaes_solver,
            'object_mesh_path': object_mesh_path,
            'object_pose_path': object_pose_path,
        }
        return self._cmaes_bundle

    def _to_cmaes_pose57(self, pose72, trans3):
        pose24 = pose72.reshape(24, 3)
        joints = [pose24[idx] for idx in self._smpl2humanoid]
        pose57 = np.concatenate([trans3, pose24[0], np.stack(joints, axis=0).reshape(-1)], axis=0)
        return pose57.astype(np.float32)

    def _from_cmaes_pose57(self, pose57, pose72):
        pose24 = pose72.reshape(24, 3).copy()
        pose24[0] = pose57[3:6]
        mapped = pose57[6:].reshape(17, 3)
        for idx, smpl_idx in enumerate(self._smpl2humanoid):
            pose24[smpl_idx] = mapped[idx]
        return pose24.reshape(-1)

    def _detect_valid_contact_frames(self, data, frame_length, batch_index=None):
        contact_valid = data.get('contact_valid') if isinstance(data, dict) else None
        if contact_valid is None:
            return list(range(frame_length))

        if torch.is_tensor(contact_valid):
            valid = contact_valid.detach()
        else:
            valid = torch.as_tensor(contact_valid)

        if valid.ndim == 3:
            valid = valid.unsqueeze(0)

        valid = valid.view(valid.shape[0], frame_length, -1)
        if batch_index is not None:
            b = int(max(0, min(int(batch_index), valid.shape[0] - 1)))
            valid = valid[b:b + 1]

        frame_has_contact = (valid > 0).any(dim=-1).any(dim=0)
        frame_ids = torch.nonzero(frame_has_contact, as_tuple=False).reshape(-1).tolist()
        return [int(fid) for fid in frame_ids]

    def _run_cmaes_physics(self, current_x, data):
        if 'obj_path' not in data or 'obj_pose' not in data:
            return current_x

        pose_end  = self.pose_dim                      # 144
        shape_end = pose_end + self.shape_dim          # 154
        trans_end = shape_end + self.trans_dim         # 157

        pose_mean_np  = self.global_mean[:pose_end].cpu().numpy()             # [144]
        pose_std_np   = self.global_std[:pose_end].cpu().numpy()              # [144]
        trans_mean_np = self.global_mean[shape_end:trans_end].cpu().numpy()   # [3]
        trans_std_np  = self.global_std[shape_end:trans_end].cpu().numpy()    # [3]

        x_np = current_x.detach().cpu().numpy()
        source_np = x_np.copy()
        optimized = x_np.copy()

        batch_size, frame_length, agent_num = x_np.shape[:3]

        obj_pose_all = data['obj_pose']
        if torch.is_tensor(obj_pose_all):
            obj_pose_all = obj_pose_all.detach().cpu().numpy()

        obj_paths = data['obj_path']
        cache_dir = os.path.join('cmaes', 'cache')
        os.makedirs(cache_dir, exist_ok=True)

        for b in range(batch_size):
            frame_candidates = self._detect_valid_contact_frames(data, frame_length, batch_index=b)
            frame_candidates = [f for f in frame_candidates if 0 <= f < frame_length]
            if not frame_candidates:
                print(f"[CMAES] batch={b}: no valid contact frames, skip physics refine")
                continue
            frame_stride = 3
            frame_candidates = frame_candidates[::frame_stride]

            object_mesh_path = obj_paths[b] if isinstance(obj_paths, list) else obj_paths
            object_pose = np.asarray(obj_pose_all[b])
            object_pose_path = os.path.join(cache_dir, f'inference_object_pose_b{b}.npy')
            np.save(object_pose_path, object_pose)

            bundle = self._get_cmaes_bundle(object_mesh_path, object_pose_path)
            env = bundle['env']
            sim_env = env.unwrapped if hasattr(env, 'unwrapped') else env
            cmaes_solver = bundle['solver']

            pose_length = object_pose.shape[0]
            sim_env.start_index = 0
            sim_env.max_frames = -1

            ext_force = np.zeros((1, agent_num, 3), dtype=np.float32)
            keyp_2d = np.zeros((agent_num, 26, 3), dtype=np.float32)
            sys_error = np.zeros((1, agent_num, 57), dtype=np.float32)
            mean = np.zeros((57 * agent_num,), dtype=np.float32)

            print(
                f"[CMAES] batch={b} start | frames={len(frame_candidates)} | times={self.cmaes_times} "
                f"| sigma={self.cmaes_sigma:.3f} | mesh={os.path.basename(str(object_mesh_path))}"
            )

            for idx, f in enumerate(tqdm(frame_candidates, desc=f"CMAES-b{b}", leave=False), start=1):
                ref_f = int(f)
                cur_f = max(0, ref_f - 1)
                sim_env.object_frame = max(0, min(ref_f, pose_length - 1))

                cur_state = np.zeros((agent_num, 114), dtype=np.float32)
                ref_state = np.zeros((1, agent_num, 114), dtype=np.float32)
                contact_points_data = data.get('contact_points')
                contact_valid_data  = data.get('contact_valid')
                contact_targets = None
                if (contact_points_data is not None and contact_valid_data is not None):
                    cp = contact_points_data
                    cv = contact_valid_data
                    if torch.is_tensor(cp):
                        cp = cp.cpu().numpy()
                    if torch.is_tensor(cv):
                        cv = cv.cpu().numpy()
                    contact_targets = []
                    for aid in range(agent_num):
                        right_pos   = cp[b, ref_f, aid, 0, :]   # [3] world frame, right wrist
                        left_pos    = cp[b, ref_f, aid, 1, :]   # [3] world frame, left wrist
                        right_valid = bool(cv[b, ref_f, aid, 0] > 0.5)
                        left_valid  = bool(cv[b, ref_f, aid, 1] > 0.5)
                        contact_targets.append((left_pos, right_pos, left_valid, right_valid))
                sim_env.contact_targets = contact_targets
                
                control_dt = 1.0 / 30.0   # 30 fps

                for aid in range(agent_num):
                    cur_pose6d_norm = source_np[b, cur_f, aid, :pose_end]
                    ref_pose6d_norm = source_np[b, ref_f, aid, :pose_end]
                    cur_pose6d = cur_pose6d_norm * pose_std_np + pose_mean_np
                    ref_pose6d = ref_pose6d_norm * pose_std_np + pose_mean_np

                    cur_pose72 = matrix_to_axis_angle(
                        rotation_6d_to_matrix(torch.from_numpy(cur_pose6d).float().reshape(-1, 6))
                    ).reshape(-1).numpy()
                    ref_pose72 = matrix_to_axis_angle(
                        rotation_6d_to_matrix(torch.from_numpy(ref_pose6d).float().reshape(-1, 6))
                    ).reshape(-1).numpy()

                    cur_trans = source_np[b, cur_f, aid, shape_end:trans_end] * trans_std_np + trans_mean_np
                    ref_trans = source_np[b, ref_f, aid, shape_end:trans_end] * trans_std_np + trans_mean_np

                    cur_pose57 = self._to_cmaes_pose57(cur_pose72, cur_trans)
                    ref_pose57 = self._to_cmaes_pose57(ref_pose72, ref_trans)

                    cur_state[aid, :57] = cur_pose57
                    ref_state[0, aid, :57] = ref_pose57
                    pose_diff = ref_pose57 - cur_pose57           # [57]
                    vel_approx = pose_diff / control_dt           # [57] 
                    cur_state[aid, 57:] = vel_approx.astype(np.float32)

                cmaes_result = cmaes_solver(
                    mean,
                    self.cmaes_sigma,
                    cur_state.copy(),
                    ref_state.copy(),
                    sys_error,
                    keyp_2d,
                    show_progress=True,
                    progress_desc=f"b{b}-f{ref_f}",
                )
                best = cmaes_result.get('best') if isinstance(cmaes_result, dict) else None
                if best is None or best.get('obs') is None:
                    print(f"[CMAES][warn] batch={b}, frame={ref_f}: solver returned no best result")
                    continue

                best_obs = np.asarray(best['obs'])
                if best_obs.ndim == 3:
                    elite_state = best_obs[0]
                else:
                    elite_state = best_obs.reshape(agent_num, -1)

                best_value = float(cmaes_result.get('best_value', np.nan))
                print(
                    f"[CMAES] batch={b} frame={ref_f} ({idx}/{len(frame_candidates)}) "
                    f"best_value={best_value:.6f}"
                )

                for aid in range(agent_num):
                    old_pose6d_norm = source_np[b, ref_f, aid, :pose_end]
                    old_pose6d = old_pose6d_norm * pose_std_np + pose_mean_np
                    old_pose72 = matrix_to_axis_angle(
                        rotation_6d_to_matrix(torch.from_numpy(old_pose6d).float().reshape(-1, 6))
                    ).reshape(-1).numpy()
                    new_pose72 = self._from_cmaes_pose57(elite_state[aid, :57], old_pose72)
                    new_pose6d_real = matrix_to_rotation_6d(
                        axis_angle_to_matrix(torch.from_numpy(new_pose72).float().reshape(-1, 3))
                    ).reshape(-1).numpy()
                    new_pose6d_norm = (new_pose6d_real - pose_mean_np) / pose_std_np
                    optimized[b, ref_f, aid, :pose_end] = new_pose6d_norm
                    optimized[b, ref_f, aid, shape_end:trans_end] = (
                        (elite_state[aid, :3] - trans_mean_np) / trans_std_np
                    )

        from scipy.ndimage import gaussian_filter1d
        arm_dims = list(range(18, 24))  + list(range(36, 42)) + \
                    list(range(54, 60))  + list(range(72, 78)) + \
                    list(range(78, 90))  + list(range(96, 120))
        for b_idx in range(optimized.shape[0]):
            for a_idx in range(optimized.shape[2]):
                seq = optimized[b_idx, :, a_idx, :]   # [T, 235]
                for d in arm_dims:
                    seq[:, d] = gaussian_filter1d(
                        seq[:, d].astype(np.float64), sigma=2.0
                    ).astype(np.float32)
                optimized[b_idx, :, a_idx, :] = seq

        bundle = self._cmaes_bundle
        if bundle is not None:
            env = bundle.get('env')
            sim_env_final = env.unwrapped if (env is not None and hasattr(env, 'unwrapped')) else env
            if sim_env_final is not None:
                sim_env_final.contact_targets = None

        return torch.from_numpy(optimized).to(device=current_x.device, dtype=current_x.dtype)

    def condition_process(self, data):
        batch_size, frame_length, _ = data['x'].shape[:3]
        device = data['x'].device

        obj_pose = data['obj_pose']
        obj_rot   = obj_pose[..., :3, :3].reshape(batch_size, frame_length, -1)
        obj_trans = obj_pose[..., :3, 3]
        object_pose_feat = torch.cat([obj_rot, obj_trans], dim=-1)
        traj_cond = self.project(object_pose_feat)

        bps_cond_raw = data['obj_bps'].reshape(batch_size, frame_length, -1)
        bps_cond = self.bps_project(bps_cond_raw)

        contact_points    = data['contact_points']
        contact_valid     = data['contact_valid'].float()
        contact_flat      = contact_points.reshape(batch_size, frame_length, -1)
        contact_valid_flat = contact_valid.reshape(batch_size, frame_length, -1)
        contact_cond = self.contact_project(
            torch.cat([contact_flat, contact_valid_flat], dim=-1)
        )

        cond = self.cond_fusion(
            torch.cat([traj_cond, bps_cond, contact_cond], dim=-1)
        )

        contact_normals      = data['contact_normals']
        contact_normals_flat = contact_normals.reshape(batch_size, frame_length, -1)

        object_condition = torch.cat([
            object_pose_feat,
            bps_cond_raw,
            contact_flat,
            contact_normals_flat,
            contact_valid_flat,
        ], dim=-1)
        obj_features = self.object_feature_proj(object_condition)

        return cond, obj_features

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def generate_noise(self, init_pose, noise=None):
        if noise is None:
            noise = torch.randn(
                (init_pose.shape[0], init_pose.shape[1], init_pose.shape[2], self.input_feats),
                device=init_pose.device, dtype=init_pose.dtype
            )
        mean = init_pose
        return noise, mean

    def input_process(self, data, img_info, mean):
        gt_pose  = data['pose_6d']
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

        mean = mean.reshape(-1, self.input_feats)

        x_a, x_b = x_t[:, :, 0], x_t[:, :, 1]
        t = t[:, 0, 0, 0]

        obj_trans = data['obj_pose'][..., :3, 3]

        shape_end = self.pose_dim + self.shape_dim
        trans_end  = shape_end + self.trans_dim

        trans_std  = self.global_std[shape_end:trans_end]
        trans_mean = self.global_mean[shape_end:trans_end]
        pose_std   = self.global_std[:6]
        pose_mean  = self.global_mean[:6]

        # Denormalize trans and root rotation to compute person-centric relative displacement
        trans_a = x_a[..., shape_end:trans_end] * trans_std + trans_mean
        trans_b = x_b[..., shape_end:trans_end] * trans_std + trans_mean

        root6d_a = x_a[..., :6] * pose_std + pose_mean
        root6d_b = x_b[..., :6] * pose_std + pose_mean

        R_a = rotation_6d_to_matrix(
            root6d_a.reshape(-1, 6)
        ).reshape(batch_size, frame_length, 3, 3)

        R_b = rotation_6d_to_matrix(
            root6d_b.reshape(-1, 6)
        ).reshape(batch_size, frame_length, 3, 3)

        rel_a = torch.matmul(
            R_a.transpose(-1, -2),
            (obj_trans - trans_a).unsqueeze(-1)
        ).squeeze(-1)

        rel_b = torch.matmul(
            R_b.transpose(-1, -2),
            (obj_trans - trans_b).unsqueeze(-1)
        ).squeeze(-1)

        x_a_aug = torch.cat([x_a, rel_a], dim=-1)
        x_b_aug = torch.cat([x_b, rel_b], dim=-1)

        cond_embed = self.feature_embed(cond)
        emb = self.time_mlp(t.reshape(-1))[:, None] + cond_embed
        emb = emb.reshape(batch_size, frame_length, -1)

        a_emb = self.motion_embed(x_a_aug)
        b_emb = self.motion_embed(x_b_aug)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)

        mask = torch.ones(batch_size, frame_length, device=x_t.device)
        key_padding_mask = ~(mask > 0.5)
        counterpart_mask = torch.ones(batch_size, frame_length, 1, device=x_t.device)

        for i, block in enumerate(self.blocks):
            h_a = block(h_a_prev, h_b_prev * counterpart_mask, emb, key_padding_mask, obj_features=obj_features)
            h_b = block(h_b_prev, h_a_prev * counterpart_mask, emb, key_padding_mask, obj_features=obj_features)
            h_a_prev = h_a
            h_b_prev = h_b

        features = torch.cat([h_a[:, :, None], h_b[:, :, None]], dim=2)
        features = features.reshape(batch_size * frame_length * agent_num, -1)
        xc = features

        pose_end  = self.pose_dim
        shape_end = pose_end + self.shape_dim
        trans_end = shape_end + self.trans_dim
        joint_end = trans_end + self.joint_dim

        pred_pose6d      = self.head(xc).view(-1, self.pose_dim)         + mean[:, :pose_end]
        pred_shape       = self.shape_head(xc).view(-1, self.shape_dim)  + mean[:, pose_end:shape_end]
        pred_cam         = self.cam_head(xc).view(-1, self.trans_dim)    + mean[:, shape_end:trans_end]
        pred_joints_flat = self.joints_head(xc).view(-1, self.joint_dim) + mean[:, trans_end:joint_end]

        pred_u_t = torch.cat([pred_pose6d, pred_shape, pred_cam, pred_joints_flat], dim=1)
        pred_u_t = pred_u_t.reshape(batch_size, frame_length, agent_num, -1)

        return pred_u_t

    def _denormalize(self, x_norm):
        return x_norm * self.global_std + self.global_mean

    def forward(self, data):
        batch_size, frame_length, agent_num = data['x'].shape[:3]
        device = data['x'].device

        data['x'] = (data['x'] - self.global_mean) / self.global_std

        cond, obj_features = self.condition_process(data)

        init_pose = torch.zeros_like(data['x'])
        noise, mean = self.generate_noise(init_pose)

        x_0 = noise
        x_1 = data['x']

        if self.training:
            t = torch.rand(batch_size, device=device)
            t = t[:, None, None, None]
            x_t = (1 - t) * x_0 + t * x_1

            u_t = x_1 - x_0

            pred_u_t = self.inference(x_t, t, cond, obj_features, data, mean)

            x_recon = x_t + (1 - t) * pred_u_t

            pose_end  = self.pose_dim
            shape_end = pose_end + self.shape_dim
            trans_end = shape_end + self.trans_dim
            joint_end = trans_end + self.joint_dim

            pred_pose6d    = x_recon[..., :pose_end].reshape(-1, self.pose_dim)
            pred_shape     = x_recon[..., pose_end:shape_end].reshape(-1, self.shape_dim)
            pred_trans     = x_recon[..., shape_end:trans_end].reshape(-1, self.trans_dim)
            pred_joints_x  = x_recon[..., trans_end:joint_end].reshape(-1, self.num_joints, 3)

            pred_joints_flat = pred_joints_x.reshape(-1, self.joint_dim)
            x = torch.cat([pred_pose6d, pred_shape, pred_trans, pred_joints_flat], dim=1)

            trans_std_g  = self.global_std[shape_end:trans_end]
            trans_mean_g = self.global_mean[shape_end:trans_end]
            pred_trans_real = pred_trans * trans_std_g + trans_mean_g

            pred = {
                'pred_x':         x,
                'pred_pose6d':    pred_pose6d,
                'pred_shape':     pred_shape,
                'pred_cam_t':     pred_trans_real,
                'pred_joints_x':  pred_joints_x,
                'u_t':            u_t,
                'pred_u_t':       pred_u_t,
            }

        else:
            x_recon = self.sample_with_condition(data, cond, obj_features, mean)
            
            pose_end  = self.pose_dim
            shape_end = pose_end + self.shape_dim
            trans_end = shape_end + self.trans_dim
            joint_end = trans_end + self.joint_dim

            pred_pose6d   = x_recon[..., :pose_end].reshape(-1, self.pose_dim)
            pred_shape    = x_recon[..., pose_end:shape_end].reshape(-1, self.shape_dim)
            pred_trans    = x_recon[..., shape_end:trans_end].reshape(-1, self.trans_dim)
            pred_joints_x = x_recon[..., trans_end:joint_end].reshape(-1, self.num_joints, 3)

            pred_rotmat = rotation_6d_to_matrix(pred_pose6d).view(-1, 24, 3, 3)
            pred_pose = matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(-1, 72)

            pred_verts, pred_joints = self.smpl(pred_shape, pred_pose, pred_trans, halpe=True)

            pred_joints_flat = pred_joints_x.reshape(-1, self.joint_dim)
            x = torch.cat([pred_pose6d, pred_shape, pred_trans, pred_joints_flat], dim=1)

            pred = {
                'pred_pose':     pred_pose,
                'pred_pose6d':   pred_pose6d,
                'pred_shape':    pred_shape,
                'pred_cam_t':    pred_trans,
                'pred_rotmat':   pred_rotmat,
                'pred_verts':    pred_verts,
                'pred_joints':   pred_joints,
                'pred_joints_x': pred_joints_x,
                'pred_x':        x,
            }

        return pred

    def sample(self, data, cond, obj_features, mean):
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

        x_norm = x_t - mean
        return self._denormalize(x_norm)

    def sample_with_condition(
        self,
        data,
        cond,
        obj_features,
        mean,
        guidance_weight=0,
        use_contact=True,
        use_temporal=True,
    ):
        batch_size, frame_length, num_agent = data['data_shape']
        device = data['pose'].device

        num_steps = self.sampling_num_steps
        integration_strength = self.integration_strength

        init_pose = torch.zeros_like(data['x'])
        x_t, mean = self.generate_noise(init_pose)

        dt = 1. / num_steps
        contact_available = use_contact
        temporal_norm = getattr(self, 'temporal_smooth_norm', 'l2')

        shape_end = self.pose_dim + self.shape_dim
        trans_end  = shape_end + self.trans_dim
        trans_std_g  = self.global_std[shape_end:trans_end]
        trans_mean_g = self.global_mean[shape_end:trans_end]

        for step in range(num_steps):
            current_x = x_t.detach()
            t = torch.ones(batch_size, device=device) * step * dt
            t = t[:, None, None, None]

            pred_u_t = self.inference(current_x, t, cond, obj_features, data, mean)

            contact_loss = None
            x_t_grad = None
            smooth_loss = None

            require_guidance = contact_available or use_temporal

            if require_guidance:
                with torch.enable_grad():
                    x_t_grad = current_x.detach().requires_grad_(True)

                    pose_end_g  = self.pose_dim
                    shape_end_g = pose_end_g + self.shape_dim
                    trans_end_g = shape_end_g + self.trans_dim

                    pose6d = x_t_grad[..., :pose_end_g]
                    shape  = x_t_grad[..., pose_end_g:shape_end_g]
                    trans  = x_t_grad[..., shape_end_g:trans_end_g]
                    pose_std_t  = self.global_std[:pose_end_g]
                    pose_mean_t = self.global_mean[:pose_end_g]
                    pose6d_real = pose6d * pose_std_t + pose_mean_t
                    pose6d_flat = pose6d_real.reshape(-1, 6)
                    rotmats = rotation_6d_to_matrix(pose6d_flat).view(-1, 24, 3, 3)

                    shape_flat = shape.reshape(-1, 10)

                    trans_flat = (trans * trans_std_g + trans_mean_g).reshape(-1, 3)

                    if contact_available:
                        pose_axis = matrix_to_axis_angle(rotmats.view(-1, 3, 3)).view(-1, 72)
                        pred_verts, pred_joints = self.smpl(shape_flat, pose_axis, trans_flat, halpe=True)

                        contact_loss = self._compute_contact_loss(pred_verts, data, pred_joints)
                        if contact_loss is not None:
                            print(f"contact_loss: {contact_loss.detach().item():.6f}")

                    if use_temporal:
                        pose6d_s = x_t_grad[..., :144]
                        trans_s  = x_t_grad[..., 154:]

                        smooth_loss = self._compute_temporal_smooth_loss_for_sampling(
                            pose6d_s, trans_s, data, norm=temporal_norm
                        )
                        if smooth_loss is not None:
                            print(f"smooth_loss: {smooth_loss.detach().item():.6f}")

            if x_t_grad is not None:
                if contact_available and contact_loss is not None:
                    grad_c = torch.autograd.grad(
                        contact_loss,
                        x_t_grad,
                        retain_graph=True,
                        create_graph=False,
                    )[0]
                    arm_joint_ids   = [13, 14, 16, 17, 18, 19, 20, 21, 22, 23]
                    torso_joint_ids = [3, 6, 9, 12]
                    arm_pose_ids   = []
                    torso_pose_ids = []

                    for j in arm_joint_ids:
                        arm_pose_ids.extend(range(j * 6, j * 6 + 6))
                    for j in torso_joint_ids:
                        torso_pose_ids.extend(range(j * 6, j * 6 + 6))

                    grad_filtered = torch.zeros_like(grad_c)
                    root_grad = grad_c[..., :6].clone()
                    root_filtered = torch.zeros_like(root_grad)
                    root_filtered[..., 0:2] = root_grad[..., 0:2]
                    grad_filtered[..., :6] = root_filtered

                    for idx in torso_pose_ids:
                        grad_filtered[..., idx] = grad_c[..., idx]
                    for idx in arm_pose_ids:
                        grad_filtered[..., idx] = grad_c[..., idx]

                    grad_filtered[..., 154] = grad_c[..., 154]
                    grad_filtered[..., 155] = grad_c[..., 155]
                    grad_filtered[..., 156] = 0.0

                    pred_u_t = pred_u_t - 200 * grad_filtered

                if use_temporal and smooth_loss is not None:
                    grad_s = torch.autograd.grad(
                        smooth_loss,
                        x_t_grad,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )[0]
                    if step < num_steps - 2:
                        smooth_weight = 400 * (step / num_steps)
                    else:
                        smooth_weight = 0.0
                    if grad_s is not None and smooth_weight > 0:
                        pred_u_t = pred_u_t - smooth_weight * grad_s

            pred_u_t = pred_u_t * integration_strength
            x_t = (current_x + pred_u_t * dt).detach()

            if self.use_cmaes_physics and step == num_steps - 2:
                x_t = self._run_cmaes_physics(x_t, data).detach()

        x_norm = x_t - mean
        return self._denormalize(x_norm)

    def _compute_temporal_smooth_loss_for_sampling(self, pose6d, trans, data, norm='l2'):
        if pose6d is None or trans is None:
            return None

        batch_size, seq_len, agent_num, _ = pose6d.shape

        if seq_len <= 1:
            return None

        pose_diff  = pose6d[:, 1:] - pose6d[:, :-1]
        trans_diff = trans[:, 1:] - trans[:, :-1]

        valid_mask = data.get('valid')
        if valid_mask is not None:
            try:
                valid_mask = valid_mask.view(batch_size, seq_len, agent_num)
                temporal_valid = (valid_mask[:, 1:] > 0.5) & (valid_mask[:, :-1] > 0.5)
                temporal_valid = temporal_valid.unsqueeze(-1)
            except RuntimeError:
                temporal_valid = None
        else:
            temporal_valid = None

        norm = (norm or 'l2').lower()
        if norm == 'l1':
            pose_term  = pose_diff.abs()
            trans_term = trans_diff.abs()
        else:
            pose_term  = pose_diff.pow(2)
            trans_term = trans_diff.pow(2)

        smooth_term = pose_term.sum(dim=-1, keepdim=True) + trans_term.sum(dim=-1, keepdim=True)

        if temporal_valid is not None:
            smooth_term = smooth_term * temporal_valid.float()
            denom = temporal_valid.float().sum()
        else:
            denom = torch.tensor(smooth_term.numel(), device=smooth_term.device, dtype=smooth_term.dtype)

        if denom.item() <= 0:
            return None

        return smooth_term.sum() / denom

    def ddim_sample_loop(self, noise, mean, cond, obj_features, img_info, data, eta=0.0):
        if obj_features is None:
            _, obj_features = self.condition_process(data)

        indices = list(range(self.num_timesteps_test))[::-1]

        img = noise
        preds = []
        for i in indices:
            t = torch.tensor([i] * noise.shape[0], device=noise.device)
            pred = self.ddim_sample(img, t, mean, cond, obj_features, data)
            preds.append(pred)

            pred_pose6d   = pred['pred_pose6d']
            pred_shape    = pred['pred_shape']
            pred_cam      = pred['pred_cam_t']
            pred_joints_x = pred['pred_joints_x']

            pred_joints_flat = pred_joints_x.reshape(pred_joints_x.shape[0], -1)
            model_output = torch.cat([pred_pose6d, pred_shape, pred_cam, pred_joints_flat], dim=-1)

            model_output = model_output.reshape(*img.shape)
            model_output = model_output - mean

            model_variance, model_log_variance = (
                self.test_posterior_variance,
                self.test_posterior_log_variance_clipped,
            )

            model_variance     = extract_into_tensor(model_variance, t, img.shape)
            model_log_variance = extract_into_tensor(model_log_variance, t, img.shape)

            pred_xstart = model_output

            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=img, t=t
            )

            assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == img.shape
            )

            eps = self._predict_eps_from_xstart(img, t, pred_xstart)

            alpha_bar      = extract_into_tensor(self.test_alphas_cumprod, t, img.shape)
            alpha_bar_prev = extract_into_tensor(self.test_alphas_cumprod_prev, t, img.shape)
            sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            )

            noise, _ = self.generate_noise(torch.zeros_like(model_output))
            mean_pred = (
                pred_xstart * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
            )
            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
            sample = mean_pred + nonzero_mask * sigma * noise

            img = sample

        return preds[-1]

    def ddim_sample(self, x, ts, mean, cond, obj_features, data):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        pred = self.inference(x, new_ts, cond, obj_features, data, mean)
        return pred

    def q_posterior_mean_variance(self, x_start, x_t, t):
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
            obj_positions = obj_poses[:, :, :3, 3]
            position_diff = torch.norm(obj_positions[:, 1:] - obj_positions[:, :-1], dim=-1)
            movement_threshold = 0.01
            movement_mask = position_diff > movement_threshold
            temporal_weights[:, 1:][movement_mask] = 1.0

        return temporal_weights

    def _compute_contact_loss(self, pred_verts, data, pred_joints):
        contact_points = data['contact_points']
        contact_valid  = data['contact_valid']
        data_shape     = data['data_shape']

        batch_size, frame_length, num_agent = data_shape
        device = pred_verts.device

        joints = pred_joints.view(batch_size, frame_length, num_agent, -1, 3)
        wrist_indices = [10, 9]
        hand_positions = torch.stack(
            [joints[..., wrist_indices[0], :], joints[..., wrist_indices[1], :]],
            dim=-2
        )

        contact_points = contact_points.to(device)
        contact_valid  = contact_valid.to(device)

        if contact_points.ndim == 4:
            contact_points = contact_points.unsqueeze(0)
            contact_valid  = contact_valid.unsqueeze(0)

        contact_points = contact_points.view(batch_size, frame_length, num_agent, 2, 3)
        contact_valid  = contact_valid.view(batch_size, frame_length, num_agent, 2)

        valid_mask = contact_valid > 0
        if not torch.any(valid_mask):
            return None

        diff = hand_positions - contact_points
        squared_dist = (diff ** 2).sum(dim=-1)
        weighted_loss = squared_dist * contact_valid
        per_instance_weight = contact_valid.sum(dim=-1)
        positive_instances = per_instance_weight > 0
        if not torch.any(positive_instances):
            return None

        normalized_loss = weighted_loss.sum(dim=-1) / per_instance_weight.clamp_min(1e-6)
        active_loss = normalized_loss * positive_instances.to(normalized_loss.dtype)
        active_count = positive_instances.sum()
        return active_loss.sum() / active_count.to(active_loss.dtype).clamp_min(1.0)


class interhuman_flow_BPS_prior(InterhumanFlow_BPS_DualDisc):
    """"""


__all__ = [
    'InterhumanFlow_BPS_DualDisc',
    'interhuman_flow_BPS_prior',
]