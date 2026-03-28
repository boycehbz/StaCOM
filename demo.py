import argparse
import os
import shutil
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "utils")))

import numpy as np
import torch
import trimesh
import cv2
from tqdm import tqdm

from model.contact_point_generator import contact_point_generator
from modules import ModelLoader
from utils.smpl_torch_batch import SMPLXModel
from utils.generated_tool import (
    _build_batch,
    render_gt_single_view_frames,
    DISTANCE_MARGIN,
    ELEVATION_DEG,
    FOV_DEG,
    AZIMUTH_DEG,
)

FRAME_LENGTH = 128
NUM_AGENTS   = 2
FPS          = 30
IMAGE_HEIGHT = 1080
IMAGE_WIDTH  = 1920


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end demo: contact generation -> motion generation -> video rendering."
    )
    parser.add_argument("--obj-mesh",     required=True,  help="Path to object mesh (.obj)")
    parser.add_argument("--obj-traj",     required=True,  help="Path to object trajectory (.npy or .npz), shape (T,4,4)")
    parser.add_argument("--affordance",   required=True,  help="Path to affordance file (.npz) containing sampled_scores")
    parser.add_argument("--contact-ckpt", required=True,  help="Path to contact_point_generator checkpoint (.pkl)")
    parser.add_argument("--motion-ckpt",  required=True,  help="Path to interhuman_flow_BPS_prior checkpoint (.pkl)")
    parser.add_argument("--body-model",   required=True,  help="Path to SMPLX_NEUTRAL.pkl")
    parser.add_argument("--output-dir",   required=True,  help="Directory to write rendered frames and output video")
    parser.add_argument("--gpu-index",    type=int, default=0)
    parser.add_argument("--physics",   action="store_true", default=False, help="Enable stability-driven physics simulation (CMA-ES)")
    return parser.parse_args()


def load_trajectory(path):
    print(f"Loading object trajectory: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path, allow_pickle=False)
        if "obj_pose" in data:
            arr = data["obj_pose"]
        else:
            key = list(data.keys())[0]
            print(f"Using key '{key}' from npz")
            arr = data[key]
    else:
        arr = np.load(path, allow_pickle=False)
    arr = arr.astype(np.float32)
    if arr.ndim == 2 and arr.shape[1] == 16:
        arr = arr.reshape(-1, 4, 4)
    assert arr.ndim == 3 and arr.shape[1:] == (4, 4), \
        f"Trajectory must be (T,4,4), got {arr.shape}"
    print(f"Trajectory shape: {arr.shape}")
    return arr


def load_affordance(path):
    print(f"Loading affordance: {path}")
    data = np.load(path, allow_pickle=False)
    assert "sampled_scores" in data, \
        f"affordance npz must contain 'sampled_scores', found keys: {list(data.keys())}"
    scores = data["sampled_scores"].astype(np.float32)
    print(f"Affordance scores shape: {scores.shape}")
    return scores


def build_contact_condition(mesh, obj_traj, affordance_scores):
    print("Building contact model condition (BPS, obj_points) ...")

    T = obj_traj.shape[0]
    n_points = int(affordance_scores.shape[0])

    np.random.seed(42)
    face_indices_sample = np.random.choice(len(mesh.faces), size=n_points, replace=True)
    sampled_points = trimesh.sample.sample_surface(mesh, n_points)[0].astype(np.float32)
    face_normals = mesh.face_normals[face_indices_sample].astype(np.float32)

    rng = np.random.default_rng(42)
    directions = rng.normal(size=(1024, 3))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-8)
    radii = rng.random(1024) ** (1.0 / 3.0)
    bps_basis = (directions * radii[:, None]).astype(np.float32)

    dense_pts = mesh.sample(4096, return_index=False).astype(np.float32)
    centroid = dense_pts.mean(axis=0, keepdims=True)
    centered = dense_pts - centroid
    max_norm = np.linalg.norm(centered, axis=1).max()
    if max_norm < 1e-6:
        max_norm = 1.0
    normalized = (centered / max_norm).astype(np.float32)

    diff = bps_basis[:, None, :] - normalized[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    bps_vec = dists.min(axis=1).astype(np.float32)

    obj_points = np.zeros((T, n_points, 7), dtype=np.float32)
    points_h = np.ones((n_points, 4), dtype=np.float32)
    points_h[:, :3] = sampled_points

    print(f"Computing world-space object points for {T} frames ...")
    for t in tqdm(range(T), desc="obj_points"):
        pose = obj_traj[t]
        world = (pose @ points_h.T).T[:, :3]
        obj_points[t] = np.concatenate([world, face_normals, affordance_scores[:, None]], axis=1)

    obj_bps = np.repeat(bps_vec[None, :], T, axis=0)

    print(f"obj_points shape: {obj_points.shape}, obj_bps shape: {obj_bps.shape}")
    return {
        "obj_pose":   obj_traj,
        "obj_points": obj_points,
        "obj_bps":    obj_bps,
    }


def run_contact_model(contact_ckpt, cond, device):
    print(f"Loading contact_point_generator from: {contact_ckpt}")
    model = contact_point_generator(smpl=None)
    ckpt = torch.load(contact_ckpt, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"contact model loaded. missing={len(missing)}, unexpected={len(unexpected)}")
    model.to(device)
    model.eval()

    print("Running contact_point_generator inference ...")
    with torch.no_grad():
        data = {
            "obj_bps":    torch.from_numpy(cond["obj_bps"]).unsqueeze(0).to(device),
            "obj_pose":   torch.from_numpy(cond["obj_pose"]).unsqueeze(0).to(device),
            "obj_points": torch.from_numpy(cond["obj_points"]).unsqueeze(0).to(device),
        }
        out = model(data)

    pred_contact_points = out["pred_contact_points"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    pred_contact_valid  = torch.sigmoid(out["pred_contact_logits"]).squeeze(0).detach().cpu().numpy().astype(np.float32)

    print(f"Contact points shape: {pred_contact_points.shape}")
    print(f"Contact valid shape:  {pred_contact_valid.shape}")
    return pred_contact_points, pred_contact_valid


def run_motion_model(motion_ckpt, obj_traj, obj_mesh_path, contact_points, contact_valid,
                     gpu_index, device, use_physics):
    print("Building motion generation batch ...")

    batch = _build_batch(
        obj_traj=obj_traj,
        num_agents=NUM_AGENTS,
        frame_length=FRAME_LENGTH,
        contact_points=contact_points,
        contact_valid=contact_valid,
        device=device,
        obj_mesh_path=os.path.abspath(obj_mesh_path),
        dtype=torch.float32,
    )
    batch["obj_path"] = os.path.abspath(obj_mesh_path)

    model_workspace = os.path.join(os.path.dirname(motion_ckpt), "demo_workspace")
    os.makedirs(model_workspace, exist_ok=True)

    config_args = {
        "mode":                           "test",
        "batchsize":                      1,
        "worker":                         0,
        "data_folder":                    "",
        "model":                          "interhuman_flow_BPS_prior",
        "use_prior":                      0,
        "train_loss":                     "",
        "test_loss":                      "",
        "model_type":                     "smplx",
        "gpu_index":                      gpu_index,
        "output":                         model_workspace,
        "lr":                             1e-4,
        "epoch":                          0,
        "use_sch":                        0,
        "pretrain":                       0,
        "pretrain_dir":                   "",
        "note":                           "",
        "viz":                            0,
        "trainset":                       "",
        "testset":                        "",
        "frame_length":                   FRAME_LENGTH,
        "amp_cmu_dir":                    "",
        "amp_use_pretrained":             0,
        "amp_discriminator_ckpt":         "",
        "interact_amp_use_pretrained":    0,
        "interact_amp_discriminator_ckpt": "",
    }

    print(f"Loading interhuman_flow_BPS_prior from: {motion_ckpt}")
    model_loader = ModelLoader(dtype=torch.float32, device=device, out_dir=model_workspace, **config_args)
    model_loader.load_checkpoint(motion_ckpt)
    model_loader.model.use_cmaes_physics = use_physics
    print(f"Physics simulation: {'ON' if use_physics else 'OFF'}")
    model_loader.model.eval()

    print("Running motion generation inference ...")
    with torch.no_grad():
        predictions = model_loader.model(batch)

    data_shape = batch["data_shape"]
    batch_size, T, n_agents = [int(v) for v in data_shape]

    pred_pose  = predictions["pred_pose"].reshape(batch_size, T, n_agents, 72)
    pred_shape = batch["betas"].reshape(batch_size, T, n_agents, -1)
    pred_trans = predictions["pred_cam_t"].reshape(batch_size, T, n_agents, 3)
    valid_mask = batch["valid"].reshape(batch_size, T, n_agents)

    print(f"pred_pose shape: {pred_pose.shape}")
    print(f"pred_trans shape: {pred_trans.shape}")

    render_sample = {
        "pose":           pred_pose[0].detach().cpu(),
        "betas":          torch.zeros_like(pred_shape[0]).detach().cpu(),
        "gt_cam_t":       pred_trans[0].detach().cpu(),
        "valid":          valid_mask[0].detach().cpu(),
        "obj_path":       os.path.abspath(obj_mesh_path),
        "obj_pose":       batch["obj_pose"][0].detach().cpu(),
        "contact_points": batch["contact_points"][0].detach().cpu(),
        "contact_valid":  batch["contact_valid"][0].detach().cpu(),
    }

    return render_sample, model_loader


def render_frames(render_sample, body_model_path, output_dir, device):
    print(f"Loading SMPLX body model from: {body_model_path}")
    smpl = SMPLXModel(device=device, model_path=body_model_path, data_type=torch.float32)

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"Rendering frames to: {frames_dir}")
    saved_paths = render_gt_single_view_frames(
        data=render_sample,
        smpl=smpl,
        output_dir=output_dir,
        frame_step=1,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        fov_deg=FOV_DEG,
        elevation_deg=ELEVATION_DEG,
        azimuth_deg=AZIMUTH_DEG,
        distance_margin=DISTANCE_MARGIN,
        background_color=(255, 255, 255),
        render_people=True,
        render_contact_points=False,
        contact_point_radius=0.0,
        gallery_offset_per_frame=(310, 0),
        gallery_frame_indices=None,
    )
    print(f"Rendered {len(saved_paths)} frame files.")
    return frames_dir


def frames_to_video(frames_dir, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(output_dir, f"res_{timestamp}.mp4")
    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.endswith(".png") and f.startswith("frame_")
    ])
    assert len(frame_files) > 0, f"No frame_*.png files found in {frames_dir}"
    print(f"Composing video from {len(frame_files)} frames -> {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (IMAGE_WIDTH, IMAGE_HEIGHT))

    for fname in tqdm(frame_files, desc="frames -> video"):
        frame_path = os.path.join(frames_dir, fname)
        img = cv2.imread(frame_path)
        assert img is not None, f"Failed to read frame: {frame_path}"
        if img.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        writer.write(img)

    writer.release()
    print(f"Video saved: {video_path}")

    shutil.rmtree(frames_dir)

    gallery_path = os.path.join(output_dir, "gallery.png")
    if os.path.exists(gallery_path):
        os.remove(gallery_path)

    return video_path


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    obj_traj   = load_trajectory(args.obj_traj)
    affordance = load_affordance(args.affordance)
    mesh       = trimesh.load(args.obj_mesh, force="mesh", process=False)
    print(f"Mesh loaded: {args.obj_mesh} | vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")

    T_raw = obj_traj.shape[0]
    if T_raw < FRAME_LENGTH:
        pad = np.zeros((FRAME_LENGTH - T_raw, 4, 4), dtype=np.float32)
        obj_traj_padded = np.concatenate([obj_traj, pad], axis=0)
        print(f"Trajectory padded from {T_raw} to {FRAME_LENGTH} frames")
    else:
        obj_traj_padded = obj_traj[:FRAME_LENGTH]
        print(f"Trajectory trimmed to {FRAME_LENGTH} frames")

    print("=" * 60)
    print("STEP 1: Build contact model condition")
    print("=" * 60)
    contact_cond = build_contact_condition(mesh, obj_traj_padded, affordance)

    print("=" * 60)
    print("STEP 2: Contact point generation")
    print("=" * 60)
    pred_contact_points, pred_contact_valid = run_contact_model(
        args.contact_ckpt, contact_cond, device
    )

    print("=" * 60)
    print("STEP 3: Motion generation")
    print("=" * 60)
    render_sample, _ = run_motion_model(
        motion_ckpt=args.motion_ckpt,
        obj_traj=obj_traj_padded,
        obj_mesh_path=args.obj_mesh,
        contact_points=pred_contact_points,
        contact_valid=pred_contact_valid,
        gpu_index=args.gpu_index,
        device=device,
        use_physics=args.physics,
    )

    print("=" * 60)
    print("STEP 4: Render frames")
    print("=" * 60)
    frames_dir = render_frames(
        render_sample=render_sample,
        body_model_path=args.body_model,
        output_dir=args.output_dir,
        device=device,
    )

    print("=" * 60)
    print("STEP 5: Compose video")
    print("=" * 60)
    video_path = frames_to_video(
        frames_dir=frames_dir,
        output_dir=args.output_dir,
    )

    print("=" * 60)
    print(f"Done. Output video: {video_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()