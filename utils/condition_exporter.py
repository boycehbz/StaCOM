import argparse
import hashlib
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shutil

import joblib
import numpy as np
import trimesh


def sample_object_points(mesh, n_points=1024, seed=42):
    np.random.seed(seed)
    points, _ = mesh.sample(n_points, return_index=True)
    return points.astype(np.float32)


def affordance_cache_path(dataset_dir, obj_path):
    obj_name = os.path.splitext(os.path.basename(obj_path))[0]
    key = hashlib.md5(obj_path.encode('utf-8')).hexdigest()[:8]
    return os.path.join(dataset_dir, 'cache', 'affordance', f'{obj_name}_{key}.npz')


def contact_cache_path(dataset_dir, obj_path, seq_index):
    obj_name = os.path.splitext(os.path.basename(obj_path))[0]
    return os.path.join(dataset_dir, 'cache', 'contacts', f'{obj_name}_seq{seq_index:05d}.npz')


def compute_affordance_from_contact_cache(dataset_dir, obj_path, sampled_points):
    mesh = trimesh.load(obj_path, force='mesh')
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if len(vertices) == 0:
        return np.zeros((sampled_points.shape[0],), dtype=np.float32)

    obj_name = os.path.splitext(os.path.basename(obj_path))[0]
    cache_dir = os.path.join(dataset_dir, 'cache', 'contacts')
    if not os.path.isdir(cache_dir):
        return np.zeros((sampled_points.shape[0],), dtype=np.float32)

    vertex_scores = np.zeros((vertices.shape[0],), dtype=np.float64)
    files = [f for f in os.listdir(cache_dir) if f.startswith(obj_name + '_seq') and f.endswith('.npz')]

    for fn in files:
        path = os.path.join(cache_dir, fn)
        try:
            with np.load(path) as data:
                if 'indices' not in data or 'valid' not in data:
                    continue
                indices = data['indices'].astype(np.int64)
                valid = data['valid'].astype(np.float64)
                mask = (indices >= 0) & (indices < vertices.shape[0]) & (valid > 0.5)
                if mask.sum() == 0:
                    continue
                np.add.at(vertex_scores, indices[mask], 1.0)
        except Exception:
            continue

    if vertex_scores.max() > 0:
        vertex_scores = vertex_scores / vertex_scores.max()

    sp = sampled_points[:, None, :]  # [N,1,3]
    vv = vertices[None, :, :]        # [1,V,3]
    d2 = ((sp - vv) ** 2).sum(axis=-1)
    nn_idx = d2.argmin(axis=1)
    return vertex_scores[nn_idx].astype(np.float32)


def read_obj_pose_sequence(dataset_dir, split, seq_index):
    annot_path = os.path.join(dataset_dir, 'annot', f'{split}.pkl')
    if not os.path.exists(annot_path):
        raise FileNotFoundError(f'annot not found: {annot_path}')

    params = joblib.load(annot_path)
    if seq_index < 0 or seq_index >= len(params):
        raise IndexError(f'seq_index out of range: {seq_index}, total={len(params)}')

    seq = params[seq_index]
    meta = seq.get('meta', {}) if isinstance(seq, dict) else {}
    obj_rel = meta.get('obj_path', '')
    if not obj_rel:
        raise ValueError('this sequence has empty obj_path in meta')

    obj_path = os.path.join(dataset_dir, obj_rel)
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f'obj not found: {obj_path}')

    frames = seq['frames'] if isinstance(seq, dict) and 'frames' in seq else seq
    obj_pose = []
    for fr in frames:
        pose = fr.get('obj_pose', np.eye(4, dtype=np.float32))
        pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)
        obj_pose.append(pose)

    obj_pose = np.stack(obj_pose, axis=0).astype(np.float32)
    return obj_path, obj_pose


def export_demo_assets(dataset_dir, split, seq_index, out_dir, n_points=1024, seed=42):
    os.makedirs(out_dir, exist_ok=True)

    obj_path, trajectory = read_obj_pose_sequence(dataset_dir, split, seq_index)

    # 1) save mesh
    mesh_name = os.path.basename(obj_path)
    out_mesh_path = os.path.join(out_dir, mesh_name)
    shutil.copyfile(obj_path, out_mesh_path)

    # 2) save trajectory
    out_traj = os.path.join(out_dir, 'trajectory.npy')
    np.save(out_traj, trajectory.astype(np.float32))

    # 3) save gt_contact
    cpath = contact_cache_path(dataset_dir, obj_path, seq_index)
    if not os.path.exists(cpath):
        raise FileNotFoundError(f'contact cache not found: {cpath}')
    with np.load(cpath) as data:
        if 'points' not in data or 'valid' not in data:
            raise ValueError(f'contact cache malformed: {cpath}')
        gt_points = data['points'].astype(np.float32)
        gt_valid = data['valid'].astype(np.float32)
    out_gt = os.path.join(out_dir, 'gt_contact.npz')
    np.savez_compressed(out_gt, points=gt_points, valid=gt_valid)

    # 4) save affordance
    apath = affordance_cache_path(dataset_dir, obj_path)
    if os.path.exists(apath):
        with np.load(apath) as data:
            if 'sampled_scores' not in data:
                raise ValueError(f'affordance cache malformed: {apath}')
            sampled_scores = data['sampled_scores'].astype(np.float32)
    else:
        mesh = trimesh.load(obj_path, force='mesh')
        sampled_points = sample_object_points(mesh, n_points=n_points, seed=seed)
        sampled_scores = compute_affordance_from_contact_cache(dataset_dir, obj_path, sampled_points)

    out_aff = os.path.join(out_dir, 'affordance.npz')
    np.savez_compressed(out_aff, sampled_scores=sampled_scores.astype(np.float32))

    return {
        'mesh_obj': out_mesh_path,
        'trajectory_npy': out_traj,
        'gt_contact_npz': out_gt,
        'affordance_npz': out_aff,
        'num_frames': int(trajectory.shape[0]),
        'num_aff_points': int(sampled_scores.shape[0]),
        'source_obj_path': obj_path,
        'source_contact_cache': cpath,
        'source_affordance_cache': apath if os.path.exists(apath) else 'computed_from_contacts',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--seq_index', type=int, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--n_points', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    info = export_demo_assets(
        dataset_dir=args.dataset_dir,
        split=args.split,
        seq_index=args.seq_index,
        out_dir=args.out_dir,
        n_points=args.n_points,
        seed=args.seed,
    )

    print('Export done:')
    for k, v in info.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()