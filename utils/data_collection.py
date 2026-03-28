import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hashlib
from collections import OrderedDict

import numpy as np
import torch
import trimesh
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from cmd_parser import parse_config
from datasets.intervae_data import HOI_Data
from utils.smpl_torch_batch import SMPLModel, SMPLXModel


def build_smpl(model_type='smpl', dtype=torch.float32):
    if model_type == 'smplx':
        return SMPLXModel(
            device=torch.device('cpu'),
            model_path='./data/SMPLX_NEUTRAL.pkl',
            data_type=dtype,
        )
    return SMPLModel(
        device=torch.device('cpu'),
        model_path='./data/SMPL_NEUTRAL.pkl',
        data_type=dtype,
    )


def parse_gender(gender_value, np_type=np.float32):
    gender = gender_value
    if isinstance(gender, np.ndarray):
        if gender.ndim == 0:
            gender = gender.item()
        elif gender.size == 1:
            gender = gender.reshape(-1)[0].item()
    if isinstance(gender, bytes):
        gender = gender.decode('utf-8')
    if isinstance(gender, str):
        gl = gender.lower()
        if gl.startswith('f') or 'female' in gl:
            return np.array([0.0], dtype=np_type)
        if gl.startswith('m') or 'male' in gl:
            return np.array([1.0], dtype=np_type)
    try:
        return np.array([float(gender)], dtype=np_type)
    except Exception:
        return np.array([-1.0], dtype=np_type)


def sample_object_points(mesh, n_points=1024, seed=42, np_type=np.float32):
    np.random.seed(seed)
    points, face_indices = mesh.sample(n_points, return_index=True)
    normals = mesh.face_normals[face_indices]
    return points.astype(np_type), normals.astype(np_type)


def build_contact_cache_path(base_dir, seq_index, obj_path):
    obj_name = os.path.splitext(os.path.basename(obj_path))[0]
    return os.path.join(base_dir, f"{obj_name}_seq{seq_index:05d}.npz")


def build_joint_cache_path(base_dir, dataset_source, seq_index):
    dataset_tag = dataset_source or 'unknown'
    return os.path.join(base_dir, f'{dataset_tag}_seq{seq_index:05d}.npz')


def build_affordance_cache_path(base_dir, obj_path):
    obj_name = os.path.splitext(os.path.basename(obj_path))[0]
    key = hashlib.md5(obj_path.encode('utf-8')).hexdigest()[:8]
    return os.path.join(base_dir, f"{obj_name}_{key}.npz")


def compute_object_motion_mask(obj_pose_seq, trans_th, rot_th):
    if obj_pose_seq is None or len(obj_pose_seq) == 0:
        return np.zeros((0,), dtype=bool)
    if obj_pose_seq.ndim != 3 or obj_pose_seq.shape[1:] != (4, 4):
        return np.ones((obj_pose_seq.shape[0],), dtype=bool)

    num_frames = obj_pose_seq.shape[0]
    if num_frames < 2:
        return np.zeros((num_frames,), dtype=bool)

    translations = obj_pose_seq[:, :3, 3]
    trans_diff = np.linalg.norm(np.diff(translations, axis=0), axis=1)

    rotations = obj_pose_seq[:, :3, :3]
    rot_delta = rotations[1:] @ np.transpose(rotations[:-1], (0, 2, 1))
    trace = np.clip((np.trace(rot_delta, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    rot_angle = np.arccos(trace)

    moving_between = (trans_diff > trans_th) | (rot_angle > rot_th)
    moving_mask = np.zeros((num_frames,), dtype=bool)
    moving_mask[:-1] |= moving_between
    moving_mask[1:] |= moving_between
    return moving_mask


def compute_sequence_contact_features(dataset, seq_index, obj_path):
    if seq_index >= len(dataset.poses):
        return None

    mesh = dataset.mesh_cache.get(obj_path)
    if mesh is None:
        mesh = trimesh.load(obj_path, force='mesh')
        dataset.mesh_cache[obj_path] = mesh

    mesh_vertices = np.asarray(mesh.vertices).astype(dataset.np_type)
    vertex_normals = np.asarray(mesh.vertex_normals).astype(dataset.np_type)
    if len(mesh_vertices) == 0:
        return None

    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(mesh_vertices)

    poses_seq = np.array(dataset.poses[seq_index], dtype=dataset.np_type)
    shapes_seq = np.array(dataset.shapes[seq_index], dtype=dataset.np_type)
    trans_seq = np.array(dataset.trans[seq_index], dtype=dataset.np_type)
    obj_pose_seq = np.array(dataset.obj_pose[seq_index], dtype=dataset.np_type)
    genders_seq = np.array(dataset.genders[seq_index], dtype=dataset.np_type)

    num_frames = poses_seq.shape[0]
    num_agents = poses_seq.shape[1] if poses_seq.ndim > 1 else 0

    contact_points = np.zeros((num_frames, dataset.max_people, 2, 3), dtype=dataset.np_type)
    contact_normals = np.zeros((num_frames, dataset.max_people, 2, 3), dtype=dataset.np_type)
    contact_valid = np.zeros((num_frames, dataset.max_people, 2), dtype=dataset.np_type)
    contact_indices = -np.ones((num_frames, dataset.max_people, 2), dtype=np.int32)
    contact_distances = np.full((num_frames, dataset.max_people, 2), np.inf, dtype=dataset.np_type)

    moving_mask = compute_object_motion_mask(
        obj_pose_seq,
        dataset.obj_motion_translation_threshold,
        dataset.obj_motion_rotation_threshold,
    )

    for agent_idx in range(min(num_agents, dataset.max_people)):
        gender_value = 0
        try:
            gender_value = int(genders_seq[0][agent_idx])
        except Exception:
            if genders_seq.ndim == 1 and agent_idx < genders_seq.shape[0]:
                gender_value = int(genders_seq[agent_idx])

        if gender_value == 0:
            smpl_model = dataset.smpl_female
        elif gender_value == 1:
            smpl_model = dataset.smpl_male
        else:
            smpl_model = dataset.smpl

        pose_tensor = torch.from_numpy(poses_seq[:, agent_idx]).reshape(-1, 72)
        shape_tensor = torch.from_numpy(shapes_seq[:, agent_idx]).reshape(-1, 10)
        trans_tensor = torch.from_numpy(trans_seq[:, agent_idx]).reshape(-1, 3)

        with torch.no_grad():
            _, joints = smpl_model(shape_tensor, pose_tensor, trans_tensor, halpe=True)

        joints_np = joints.detach().cpu().numpy()[..., :3]
        hand_positions = np.stack([joints_np[:, 9], joints_np[:, 10]], axis=1)

        for frame_idx in range(num_frames):
            if not moving_mask[frame_idx]:
                continue
            obj_pose = obj_pose_seq[frame_idx]
            if obj_pose.shape != (4, 4):
                continue
            obj_pose_inv = np.linalg.inv(obj_pose)
            rot_mat = obj_pose[:3, :3]

            for hand_idx in range(2):
                hand_pos_world = hand_positions[frame_idx, hand_idx]
                if not np.isfinite(hand_pos_world).all():
                    continue

                hand_local_h = np.append(hand_pos_world, 1.0)
                hand_local = obj_pose_inv.dot(hand_local_h)[:3]

                distances, indices = nbrs.kneighbors(hand_local[None, :], return_distance=True)
                distance = distances[0, 0]
                nearest_index = int(indices[0, 0])

                nearest_point_local = mesh_vertices[nearest_index]
                nearest_normal_local = vertex_normals[nearest_index]
                nearest_point_world = rot_mat.dot(nearest_point_local) + obj_pose[:3, 3]
                nearest_normal_world = rot_mat.dot(nearest_normal_local)

                contact_points[frame_idx, agent_idx, hand_idx] = nearest_point_world
                contact_normals[frame_idx, agent_idx, hand_idx] = nearest_normal_world
                contact_indices[frame_idx, agent_idx, hand_idx] = nearest_index
                contact_distances[frame_idx, agent_idx, hand_idx] = distance

                if distance <= dataset.contact_detection_threshold:
                    contact_valid[frame_idx, agent_idx, hand_idx] = 1.0

    return {
        'points': contact_points.astype(dataset.np_type),
        'normals': contact_normals.astype(dataset.np_type),
        'valid': contact_valid.astype(dataset.np_type),
        'indices': contact_indices.astype(np.int32),
        'distances': contact_distances.astype(dataset.np_type),
    }


def compute_sequence_joints(dataset, seq_index):
    poses_seq = np.array(dataset.poses[seq_index], dtype=dataset.np_type)
    shapes_seq = np.array(dataset.shapes[seq_index], dtype=dataset.np_type)
    trans_seq = np.array(dataset.trans[seq_index], dtype=dataset.np_type)
    genders_seq = np.array(dataset.genders[seq_index], dtype=dataset.np_type)

    num_frames = poses_seq.shape[0]
    joints_seq = np.zeros((num_frames, dataset.max_people, dataset.num_joints, 3), dtype=dataset.np_type)

    for agent_idx in range(dataset.max_people):
        if genders_seq.ndim > 1 and agent_idx < genders_seq.shape[1]:
            gender_value = genders_seq[0][agent_idx]
        elif genders_seq.ndim == 1 and agent_idx < genders_seq.shape[0]:
            gender_value = genders_seq[agent_idx]
        else:
            gender_value = -1

        gender_value = float(parse_gender(gender_value, dataset.np_type)[0])

        if gender_value == 0:
            smpl_model = dataset.smpl_female
        elif gender_value == 1:
            smpl_model = dataset.smpl_male
        else:
            smpl_model = dataset.smpl

        pose_tensor = torch.from_numpy(poses_seq[:, agent_idx]).reshape(-1, 72)
        shape_tensor = torch.from_numpy(shapes_seq[:, agent_idx]).reshape(-1, 10)
        trans_tensor = torch.from_numpy(trans_seq[:, agent_idx]).reshape(-1, 3)

        with torch.no_grad():
            _, joints = smpl_model(shape_tensor, pose_tensor, trans_tensor, halpe=True)

        joints_seq[:, agent_idx] = joints.detach().cpu().numpy()[..., :3]

    return joints_seq.astype(dataset.np_type)


def compute_object_affordance_from_contacts(dataset, obj_path, sampled_points):
    mesh = dataset.mesh_cache.get(obj_path)
    if mesh is None:
        mesh = trimesh.load(obj_path, force='mesh')
        dataset.mesh_cache[obj_path] = mesh

    vertices = np.asarray(mesh.vertices).astype(dataset.np_type)
    if len(vertices) == 0:
        return np.zeros((sampled_points.shape[0],), dtype=dataset.np_type)

    obj_name = os.path.splitext(os.path.basename(obj_path))[0]
    if not os.path.isdir(dataset.contact_cache_dir):
        return np.zeros((sampled_points.shape[0],), dtype=dataset.np_type)

    vertex_scores = np.zeros((vertices.shape[0],), dtype=np.float64)
    matched_files = [
        f for f in os.listdir(dataset.contact_cache_dir)
        if f.startswith(obj_name + '_seq') and f.endswith('.npz')
    ]

    for fn in matched_files:
        cache_path = os.path.join(dataset.contact_cache_dir, fn)
        try:
            with np.load(cache_path) as cached:
                if 'indices' not in cached or 'valid' not in cached:
                    continue
                indices = cached['indices'].astype(np.int64)
                valid = cached['valid'].astype(np.float64)
                mask = (indices >= 0) & (indices < vertices.shape[0]) & (valid > 0.5)
                if mask.sum() == 0:
                    continue
                np.add.at(vertex_scores, indices[mask], 1.0)
        except Exception:
            continue

    if vertex_scores.max() > 0:
        vertex_scores = vertex_scores / vertex_scores.max()

    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(vertices)
    _, nn_idx = nbrs.kneighbors(sampled_points.astype(dataset.np_type))
    sampled_scores = vertex_scores[nn_idx[:, 0]].astype(dataset.np_type)
    return sampled_scores


def collect_for_dataset(dataset):
    print(f"\n[Collect] dataset_dir={dataset.dataset_dir} source={getattr(dataset, 'dataset_source', 'unknown')} sequences={len(dataset.poses)}")

    unique_obj_paths = OrderedDict()
    for p in getattr(dataset, 'obj_path', []):
        if p:
            unique_obj_paths[p] = True

    for seq_index in tqdm(range(len(dataset.poses)), desc='Contact/Joints cache'):
        obj_path = dataset.obj_path[seq_index] if seq_index < len(dataset.obj_path) else ''

        if obj_path:
            contact_data = compute_sequence_contact_features(dataset, seq_index, obj_path)
            if contact_data is not None:
                cpath = build_contact_cache_path(dataset.contact_cache_dir, seq_index, obj_path)
                os.makedirs(os.path.dirname(cpath), exist_ok=True)
                np.savez_compressed(
                    cpath,
                    points=contact_data['points'],
                    normals=contact_data['normals'],
                    valid=contact_data['valid'],
                    indices=contact_data['indices'],
                    distances=contact_data['distances'],
                )

        joints_seq = compute_sequence_joints(dataset, seq_index)
        jpath = build_joint_cache_path(dataset.joint_cache_dir, getattr(dataset, 'dataset_source', 'unknown'), seq_index)
        os.makedirs(os.path.dirname(jpath), exist_ok=True)
        np.savez_compressed(jpath, joints=joints_seq.astype(dataset.np_type))

    for obj_path in tqdm(list(unique_obj_paths.keys()), desc='Affordance cache', leave=False):
        mesh = dataset.mesh_cache.get(obj_path)
        if mesh is None:
            mesh = trimesh.load(obj_path, force='mesh')
            dataset.mesh_cache[obj_path] = mesh
        sampled_points, _ = sample_object_points(mesh, n_points=1024, seed=42, np_type=dataset.np_type)
        sampled_scores = compute_object_affordance_from_contacts(dataset, obj_path, sampled_points)

        apath = build_affordance_cache_path(dataset.affordance_cache_dir, obj_path)
        os.makedirs(os.path.dirname(apath), exist_ok=True)
        np.savez_compressed(apath, sampled_scores=sampled_scores.astype(dataset.np_type))


def main():
    args = parse_config()

    dtype = torch.float32
    smpl = build_smpl(args.get('model_type', 'smpl'), dtype=dtype)
    data_folder = args.get('data_folder', '')

    dataset_specs = []
    for name in args.get('trainset', '').split(' '):
        name = name.strip()
        if name:
            dataset_specs.append((name, True))
    for name in args.get('testset', '').split(' '):
        name = name.strip()
        if name:
            dataset_specs.append((name, False))

    seen = OrderedDict()
    for spec in dataset_specs:
        seen[spec] = True

    for name, is_train in seen.keys():
        dataset = HOI_Data(
            train=is_train,
            dtype=dtype,
            data_folder=data_folder,
            name=name,
            smpl=smpl,
            frame_length=args.get('frame_length', 16),
        )
        collect_for_dataset(dataset)

    print('\nDone. All caches have been recomputed and overwritten.')


if __name__ == '__main__':
    main()