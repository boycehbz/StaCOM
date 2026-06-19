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

from utils.video_processing import generate_mp4
from utils.imutils import (
    build_contact_condition,
    load_affordance,
    load_trajectory,
    render_frames,
    run_contact_model,
    run_motion_model,
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["PYOPENGL_PLATFORM"] = "egl" #osmesa egl

# sys.argv = [
#     '',
#     '--obj-mesh=data/test/01/box001.obj',
#     '--obj-traj=data/test/01/trajectory.npy',
#     '--affordance=data/test/01/affordance.npz',
#     '--contact-ckpt=data/contact_epoch200.pkl',
#     '--motion-ckpt=data/hoi_epoch020.pkl',
#     '--body-model=data/SMPLX_NEUTRAL.pkl',
#     '--output-dir=output/',
# ]


FRAME_LENGTH = 128
FPS          = 30


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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(args.output_dir, f"res_{timestamp}.mp4")
    print(f"Composing video -> {video_path}")
    generate_mp4(frames_dir, video_path, output_fps=FPS)
    print(f"Video saved: {video_path}")

    shutil.rmtree(frames_dir)

    gallery_path = os.path.join(args.output_dir, "gallery.png")
    if os.path.exists(gallery_path):
        os.remove(gallery_path)

    print("=" * 60)
    print(f"Done. Output video: {video_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
