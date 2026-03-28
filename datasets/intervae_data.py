import os
import hashlib
import torch
import numpy as np
import cv2
import trimesh
import sys
import joblib
sys.path.append('./')
from utils.geometry import estimate_translation_np
from utils.imutils import get_crop, keyp_crop2origin, surface_projection, img_crop2origin
from datasets.base import base
from utils.rotation_conversions import *
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class HOI_Data(base):
    def __init__(self, train=True, dtype=torch.float32, data_folder='', name='', smpl=None, frame_length=16):
        super(HOI_Data, self).__init__(train=train, dtype=dtype, data_folder=data_folder, name=name, smpl=smpl)
        # dataset_name, subset_options = self._parse_dataset_spec(name)
        # self.dataset_subset_options = subset_options
        # self.dataset_requested_name = name
        # super(HOI_Data, self).__init__(train=train, dtype=dtype, data_folder=data_folder, name=dataset_name, smpl=smpl)

        self.max_people = 2
        self.max_length = 128
        self.dataset_name = name
        self.joint_dataset = ['Panoptic', 'JTA']
        self.obj_meshes = []
        self.n_bps_points = 1024
        self.bps_surface_sample_points = 4096
        self.bps_random_seed = 42
        self.bps_basis = self._generate_bps_basis(self.n_bps_points, self.bps_random_seed)
        # self.mesh_cache = {}  
        if self.is_train:
            dataset_annot = os.path.join(self.dataset_dir, 'annot/train.pkl')
            self.eval = False
        else:
            self.eval = True
            dataset_annot = os.path.join(self.dataset_dir, 'annot/test.pkl')

        self.poses, self.shapes, self.trans, self.genders, self.obj_pose, self.obj_path = [], [], [], [], [], []
        self.mesh_cache = {}
        self.person_valids = []
        self.obj_points_cache = {}
        self.contact_cache = {}
        self.joints_cache = {}
        self.affordance_cache = {}
        self.contact_cache_dir = os.path.join(self.dataset_dir, 'cache', 'contacts')
        self.affordance_cache_dir = os.path.join(self.dataset_dir, 'cache', 'affordance')
        self.joint_cache_dir = os.path.join(self.dataset_dir, 'cache', 'joints')
        self.obj_points_cache = {}
        os.makedirs(self.contact_cache_dir, exist_ok=True)
        os.makedirs(self.joint_cache_dir, exist_ok=True)
        os.makedirs(self.affordance_cache_dir, exist_ok=True)
        self.contact_detection_threshold = 0.2
        self.obj_motion_translation_threshold = 1e-3
        self.obj_motion_rotation_threshold = np.deg2rad(1.0)
        self.num_joints = 26

        if os.path.exists(dataset_annot):
            self.has_object_data = True
            self.contact_cache_dir = os.path.join(self.dataset_dir, 'cache', 'contacts')
            os.makedirs(self.contact_cache_dir, exist_ok=True)
            self._load_core4d_sequences(dataset_annot)
        else:
            print("No Data Annot!")
        
        self.dataset_source = 'core4d'
        self.iter_list = []
        core4d_window_count = 0
        for i in range(len(self.poses)):
            if self.is_train:
                stride = 1
                if len(self.poses[i]) <= self.max_length:
                    self.iter_list.append([i, 0])
                    core4d_window_count += 1
                else:
                    for n in range(0, (len(self.poses[i]) - self.max_length), stride):
                        self.iter_list.append([i, n])
                        core4d_window_count += 1
            else:
                self.iter_list.append([i, 0])

        self.len = len(self.iter_list)

        if self.is_train:
            # print(f'[Data] Core4D windows: {core4d_window_count}, InterX windows: {interx_window_count}')
            print(
                f'[Data] Core4D windows: {core4d_window_count}, '
            )

        # self._init_condition_visualization()

    def _pad_frame_values(self, values, value_dim):
        frame = np.zeros((self.max_people, value_dim), dtype=self.np_type)
        valid_mask = np.zeros((self.max_people,), dtype=self.np_type)
        count = min(len(values), self.max_people)
        for idx in range(count):
            frame[idx] = values[idx]
            valid_mask[idx] = 1.0
        return frame, valid_mask

    def _parse_gender(self, gender_value):
        gender = gender_value
        if isinstance(gender, np.ndarray):
            if gender.ndim == 0:
                gender = gender.item()
            elif gender.size == 1:
                gender = gender.reshape(-1)[0].item()
        if isinstance(gender, bytes):
            gender = gender.decode('utf-8')
        if isinstance(gender, str):
            gender_lower = gender.lower()
            if gender_lower.startswith('f') or 'female' in gender_lower:
                return np.array([0.0], dtype=self.np_type)
            if gender_lower.startswith('m') or 'male' in gender_lower:
                return np.array([1.0], dtype=self.np_type)
        try:
            return np.array([float(gender)], dtype=self.np_type)
        except Exception:
            return np.array([-1.0], dtype=self.np_type)

    def _load_core4d_sequences(self, annot_path):
        params = self.load_pkl(annot_path)
        for seq in tqdm(params, total=len(params)):
            if len(seq) < 1:
                continue
            meta = seq.get('meta', {}) if isinstance(seq, dict) else {}
            obj_rel_path = meta.get('obj_path', '')
            obj_path = os.path.join(self.dataset_dir, obj_rel_path) if obj_rel_path else ''
            if obj_path and obj_path not in self.mesh_cache:
                mesh = trimesh.load(obj_path)
                self.mesh_cache[obj_path] = mesh
                points, normals = self.sample_object_points(mesh)
                bps_descriptor = self.compute_object_bps(mesh)
                affordance_points = self._load_or_compute_object_affordance(obj_path, points)
                self.obj_points_cache[obj_path] = {
                    'points': points,
                    'normals': normals,
                    'bps': bps_descriptor,
                    'affordance': affordance_points,
                }
            seq_poses, seq_shapes, seq_trans = [], [], []
            seq_genders, seq_obj_pose, seq_valids = [], [], []

            frames = seq['frames'] if isinstance(seq, dict) else seq
            for frame in frames:
                frame_obj_pose = frame.get('obj_pose', np.eye(4, dtype=self.np_type))
                pose_list, shape_list, trans_list, gender_list = [], [], [], []

                for key, value in frame.items():
                    if key in ['img_path', 'h_w', 'obj_pose']:
                        continue

                    pose_list.append(np.array(value['pose'], dtype=self.np_type).reshape(72,))
                    shape_list.append(np.array(value['betas'], dtype=self.np_type).reshape(10,))
                    trans_list.append(np.array(value['trans'], dtype=self.np_type).reshape(3,))
                    gender_list.append(self._parse_gender(value.get('gender', -1)))
                if len(pose_list) == 0:
                    continue

                pose_frame, valid_mask = self._pad_frame_values(pose_list, 72)
                shape_frame, _ = self._pad_frame_values(shape_list, 10)
                trans_frame, _ = self._pad_frame_values(trans_list, 3)
                gender_frame, _ = self._pad_frame_values(gender_list, 1)

                seq_poses.append(pose_frame)
                seq_shapes.append(shape_frame)
                seq_trans.append(trans_frame)
                seq_genders.append(gender_frame[:, 0])
                seq_obj_pose.append(np.array(frame_obj_pose, dtype=self.np_type).reshape(4, 4))
                seq_valids.append(valid_mask)

            if len(seq_poses) == 0:
                continue

            self.obj_path.append(obj_path)
            if obj_path:
                self.obj_meshes.append(self.mesh_cache[obj_path])
            else:
                self.obj_meshes.append(None)

            self.poses.append(np.array(seq_poses, dtype=self.np_type))
            self.shapes.append(np.array(seq_shapes, dtype=self.np_type))
            self.trans.append(np.array(seq_trans, dtype=self.np_type))
            self.genders.append(np.array(seq_genders, dtype=self.np_type))
            self.obj_pose.append(np.array(seq_obj_pose, dtype=self.np_type))
            self.person_valids.append(np.array(seq_valids, dtype=self.np_type))

        del params

    def sample_object_points(self, mesh, n_points=1024, seed=42):
        np.random.seed(seed)
        points, face_indices = mesh.sample(n_points, return_index=True)
        normals = mesh.face_normals[face_indices]
        return points.astype(self.np_type), normals.astype(self.np_type)
    

    def _generate_bps_basis(self, n_points, seed=42):
        rng = np.random.default_rng(seed)
        directions = rng.normal(size=(n_points, 3))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.maximum(norms, 1e-8)
        radii = rng.random(n_points) ** (1.0 / 3.0)
        basis = directions * radii[:, None]
        return basis.astype(self.np_type)

    def _normalize_points(self, points):
        centroid = np.mean(points, axis=0, keepdims=True)
        centered = points - centroid
        max_norm = np.linalg.norm(centered, axis=1).max()
        if max_norm < 1e-6:
            max_norm = 1.0
        normalized = centered / max_norm
        return normalized.astype(self.np_type)

    def _compute_bps_descriptor(self, points):
        normalized_points = self._normalize_points(points)
        nbrs = NearestNeighbors(n_neighbors=1)
        nbrs.fit(normalized_points)
        distances, _ = nbrs.kneighbors(self.bps_basis)
        return distances.squeeze(-1).astype(self.np_type)

    def compute_object_bps(self, mesh):
        dense_points = mesh.sample(self.bps_surface_sample_points, return_index=False)
        dense_points = dense_points.astype(self.np_type)
        return self._compute_bps_descriptor(dense_points)
    
    def transform_points_to_world(self, points, obj_pose):
        points_homo = np.ones((points.shape[0], 4), dtype=self.np_type)
        points_homo[:, :3] = points
        transformed_points = np.dot(obj_pose, points_homo.T).T
        return transformed_points[:, :3]
    
    # Data preprocess
    def create_data(self, index=0):
        
        load_data = {}
        
        seq_ind, start    = self.iter_list[index]
        
        obj_path          = self.obj_path[seq_ind]
        valid_poses       = np.array(self.poses[seq_ind], dtype=self.np_type)[start:start+self.max_length]
        valid_shapes      = np.array(self.shapes[seq_ind], dtype=self.np_type)[start:start+self.max_length]
        valid_trans       = np.array(self.trans[seq_ind], dtype=self.np_type)[start:start+self.max_length]
        valid_obj_poses   = np.array(self.obj_pose[seq_ind], dtype=self.np_type)[start:start+self.max_length]
        genders           = np.array(self.genders[seq_ind], dtype=self.np_type)[0]
        valid_len         = len(valid_poses)
        

        poses                         = np.zeros((self.max_length, self.max_people, 72), dtype=self.np_type)
        shapes                        = np.zeros((self.max_length, self.max_people, 10), dtype=self.np_type)
        trans                         = np.zeros((self.max_length, self.max_people, 3), dtype=self.np_type)
        obj_poses                     = np.zeros((self.max_length, 4, 4), dtype=self.np_type)
        valids                        = np.zeros((self.max_length, self.max_people), dtype=self.np_type)
        # obj_points                    = np.zeros((self.max_length, 100, 6), dtype=self.np_type)
        obj_points                    = np.zeros((self.max_length, 1024, 7), dtype=self.np_type)  # xyz + normal + affordance
        obj_bps                       = np.zeros((self.max_length, self.n_bps_points), dtype=self.np_type) # n_bps_points = 1024
        contact_points                = np.zeros((self.max_length, self.max_people, 2, 3), dtype=self.np_type)
        contact_normals               = np.zeros((self.max_length, self.max_people, 2, 3), dtype=self.np_type)
        contact_valid                 = np.zeros((self.max_length, self.max_people, 2), dtype=self.np_type)

        poses[:valid_len] = valid_poses
        shapes[:valid_len] = valid_shapes
        trans[:valid_len] = valid_trans
        obj_poses[:valid_len] = valid_obj_poses

        seq_valid_mask = None
        if seq_ind < len(self.person_valids):
            seq_valid_mask = np.array(self.person_valids[seq_ind], dtype=self.np_type)
        if seq_valid_mask is not None and seq_valid_mask.size > 0:
            valids[:valid_len] = seq_valid_mask[:valid_len]
        else:
            valids[:valid_len] = 1

        obj_path = ''
        has_object = False
        if seq_ind < len(self.obj_path):
            obj_path = self.obj_path[seq_ind]
            # has_object = bool(obj_path)

        cached_data = self.obj_points_cache.get(obj_path)
        mesh = self.mesh_cache[obj_path]

        points, normals = self.sample_object_points(mesh)
        bps_descriptor = self.compute_object_bps(mesh)
        affordance_points = self._load_or_compute_object_affordance(obj_path, points)
        cached_data = {
            'points': points,
            'normals': normals,
            'bps': bps_descriptor,
            'affordance': affordance_points,
        }

        self.obj_points_cache[obj_path] = cached_data
        
        local_points = cached_data['points']
        local_normals = cached_data['normals']
        local_affordance = cached_data.get('affordance', np.zeros((local_points.shape[0],), dtype=self.np_type))
        bps_descriptor = cached_data['bps']

        obj_bps[:valid_len] = bps_descriptor[None, :]

        contact_data = self.contact_cache.get((seq_ind, obj_path), None)

        if contact_data is None:
            contact_data = self._load_or_compute_contact_features(seq_ind, obj_path)

        # print(contact_data.keys())
        contact_points_seq = contact_data['points'][start:start+self.max_length]
        contact_normals_seq = contact_data['normals'][start:start+self.max_length]
        contact_valid_seq = contact_data['valid'][start:start+self.max_length]

        contact_points[:valid_len] = contact_points_seq[:valid_len]
        contact_normals[:valid_len] = contact_normals_seq[:valid_len]
        contact_valid[:valid_len] = contact_valid_seq[:valid_len]

        for frame_idx in range(valid_len):
            world_points = self.transform_points_to_world(local_points, valid_obj_poses[frame_idx])
            # obj_points[frame_idx] = np.concatenate([world_points, local_normals], axis=1)
            obj_points[frame_idx] = np.concatenate([world_points, local_normals, local_affordance[:, None]], axis=1)
            del world_points
        
        joints_seq = self._get_sequence_joints(seq_ind)
        joints_window = np.zeros((self.max_length, self.max_people, self.num_joints, 3), dtype=self.np_type)
        joints_window[:valid_len] = joints_seq[start:start+self.max_length][:valid_len]

        joint_conf = valids[..., None, None].astype(self.np_type)
        joint_conf = np.repeat(joint_conf, self.num_joints, axis=2)

        gt_joints = np.zeros((self.max_length, self.max_people, self.num_joints, 4), dtype=self.np_type)
        gt_joints[..., :3] = joints_window
        gt_joints[..., 3:4] = joint_conf

        load_data['gt_joints'] = torch.from_numpy(gt_joints)

        if not self.is_train:
            # vertss, jointss = [], []
            vertss = []
            for i in range(self.max_people):
                gender = genders[i]
                if gender == 0:
                    smpl_model = self.smpl_female
                elif gender == 1:
                    smpl_model = self.smpl_male
                else:
                    smpl_model = self.smpl

                temp_pose = torch.from_numpy(poses[:, i]).reshape(-1, 72).contiguous()
                temp_shape = torch.from_numpy(shapes[:, i]).reshape(-1, 10).contiguous()
                temp_trans = torch.from_numpy(trans[:, i]).reshape(-1, 3).contiguous()
                verts, _ = smpl_model(temp_shape, temp_pose, temp_trans, halpe=True)

                vertss.append(verts[:, None])

            vertss = torch.cat(vertss, dim=1)
            # jointss = torch.cat(jointss, dim=1)

            vertss = vertss.reshape(self.max_length, self.max_people, -1, 3)
            # gt_joints = jointss.reshape(self.max_length, self.max_people, 26, 4)

            load_data['verts'] = vertss
            # load_data['gt_joints'] = gt_joints


        has_3d = np.ones((self.max_length, self.max_people), dtype=self.np_type)
        has_smpls = np.ones((self.max_length, self.max_people), dtype=self.np_type)

        gt_trans = trans.reshape(self.max_length, self.max_people, 3)
        poses = torch.from_numpy(poses)

        pose_6d = poses.reshape(-1, 3)
        pose_6d = axis_angle_to_matrix(pose_6d)
        pose_6d = matrix_to_rotation_6d(pose_6d)
        pose_6d = pose_6d.reshape(self.max_length, self.max_people, -1)

        imgnames = ['seq%04d_frame%05d.jpg' %(seq_ind, i+start) for i in range(self.max_length)]

        load_data['valid'] = valids
        load_data['has_3d'] = has_3d
        load_data['has_smpl'] = has_smpls
        # load_data['img'] = self.normalize_img(img)
        load_data['pose'] = poses
        load_data['pose_6d'] = pose_6d
        load_data['obj_pose'] = obj_poses
        load_data['betas'] = torch.from_numpy(shapes)
        load_data['gt_cam_t'] = torch.from_numpy(gt_trans)
        load_data['imgname'] = imgnames
        load_data['obj_path'] = obj_path
        # domain_label = 0.0 if self.has_object_data else 1.0
        domain_label = 1.0 if self.dataset_source in ('interx', 'omomo') else 0.0
        load_data['dataset_domain'] = torch.full((self.max_length, 1), domain_label, dtype=torch.float32)

        # x = torch.cat([load_data['pose_6d'], load_data['betas'], load_data['gt_cam_t']], dim=-1)
        joints_flat = torch.from_numpy(joints_window).reshape(self.max_length, self.max_people, -1)
        load_data['joints'] = torch.from_numpy(joints_window)

        x = torch.cat(
            [load_data['pose_6d'], load_data['betas'], load_data['gt_cam_t'], joints_flat],
            dim=-1,
        )
        load_data['x'] = x
        # load_data['obj_points'] = torch.from_numpy(obj_points)
        # load_data['obj_normals'] = torch.from_numpy(obj_normals)
        load_data['obj_affordance'] = torch.from_numpy(obj_points[..., 6])
        load_data['obj_points'] = torch.from_numpy(obj_points)
        load_data['obj_bps'] = torch.from_numpy(obj_bps)
        load_data['contact_points'] = torch.from_numpy(contact_points)
        load_data['contact_normals'] = torch.from_numpy(contact_normals)
        load_data['contact_valid'] = torch.from_numpy(contact_valid)

        return load_data

    def __getitem__(self, index):
        data = self.create_data(index)
        # self._visualize_sample(index, data)
        return data

    def _build_contact_cache_path(self, base_dir, seq_index, obj_path):
        obj_name = os.path.splitext(os.path.basename(obj_path))[0]
        return os.path.join(base_dir, f"{obj_name}_seq{seq_index:05d}.npz")

    def _load_contact_features_generic(self, seq_index, obj_path, cache_dir, cache_store):
        cache_key = (seq_index, obj_path)
        if cache_key in cache_store:
            return cache_store[cache_key]
        cache_path = self._build_contact_cache_path(cache_dir, seq_index, obj_path)
        if not cache_path or (not os.path.exists(cache_path)):
            return None
        try:
            with np.load(cache_path) as cached:
                required_keys = {'points', 'normals', 'valid'}
                if not required_keys.issubset(cached.files):
                    return None
                contact_data = {
                    'points': cached['points'].astype(self.np_type),
                    'normals': cached['normals'].astype(self.np_type),
                    'valid': cached['valid'].astype(self.np_type),
                    'indices': cached['indices'].astype(np.int32) if 'indices' in cached.files else None,
                    'distances': cached['distances'].astype(self.np_type) if 'distances' in cached.files else None,
                }
        except Exception:
            return None

        cache_store[cache_key] = contact_data
        return contact_data

    def _load_or_compute_contact_features(self, seq_index, obj_path):
        return self._load_contact_features_generic(
            seq_index,
            obj_path,
            self.contact_cache_dir,
            self.contact_cache
        )

    def _get_sequence_joints(self, seq_index):
        if seq_index in self.joints_cache:
            return self.joints_cache[seq_index]
        
        cache_path = self._joint_cache_path(seq_index)
        cached_joints = self._load_joint_cache(cache_path)
        if cached_joints is None:
            return None
        
        self.joints_cache[seq_index] = cached_joints
        return cached_joints
    
    def _joint_cache_path(self, seq_index):
        dataset_tag = getattr(self, 'dataset_source', None) or 'unknown'
        return os.path.join(self.joint_cache_dir, f'{dataset_tag}_seq{seq_index:05d}.npz')

    def _load_joint_cache(self, cache_path):
        if not cache_path or not os.path.exists(cache_path):
            return None
        try:
            with np.load(cache_path) as cached:
                if 'joints' not in cached:
                    return None
                return cached['joints'].astype(self.np_type)
        except Exception:
            return None
    
    def _build_object_affordance_cache_path(self, obj_path):
        obj_name = os.path.splitext(os.path.basename(obj_path))[0]
        key = hashlib.md5(obj_path.encode('utf-8')).hexdigest()[:8]
        return os.path.join(self.affordance_cache_dir, f"{obj_name}_{key}.npz")

    def _load_or_compute_object_affordance(self, obj_path, sampled_points):
        if not obj_path:
            return np.zeros((sampled_points.shape[0],), dtype=self.np_type)

        cache_key = obj_path
        if cache_key in self.affordance_cache:
            cached = self.affordance_cache[cache_key]
            if cached.shape[0] == sampled_points.shape[0]:
                return cached

        cache_path = self._build_object_affordance_cache_path(obj_path)
        if not cache_path or (not os.path.exists(cache_path)):
            return np.zeros((sampled_points.shape[0],), dtype=self.np_type)

        try:
            with np.load(cache_path) as cached:
                if 'sampled_scores' not in cached.files:
                    return np.zeros((sampled_points.shape[0],), dtype=self.np_type)
                sampled_scores = cached['sampled_scores'].astype(self.np_type)
        except Exception:
            return np.zeros((sampled_points.shape[0],), dtype=self.np_type)

        if sampled_scores.shape[0] != sampled_points.shape[0]:
            return np.zeros((sampled_points.shape[0],), dtype=self.np_type)

        self.affordance_cache[cache_key] = sampled_scores
        return sampled_scores

    def __len__(self):
        return self.len


if __name__ == '__main__':
    from utils.smpl_torch_batch import SMPLModel
    from utils.FileLoaders import save_pkl
    from tqdm import tqdm

    smpl = SMPLModel(model_path='data/SMPL_NEUTRAL.pkl')

    dataset = InterVAE_Data(train=True, data_folder='', name='InterHuman', smpl=smpl, frame_length=16)


    for i, data in tqdm(enumerate(dataset), total=dataset.len):
    
        save_pkl('InterHuman_generated/train/%06d.pkl' %i, data['x'])