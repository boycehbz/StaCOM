import os
import io
import base64
import traceback
import tempfile
from typing import Tuple

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from flask import Flask, request, jsonify, render_template_string

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from model.contact_point_generator import contact_point_generator


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024


# --------------------------
# Basic utils
# --------------------------
def fig_to_b64(fig, fmt="png", dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def bytes_to_b64(data: bytes):
    return base64.b64encode(data).decode()


def b64_to_bytes(data: str):
    return base64.b64decode(data)


# --------------------------
# BPS
# --------------------------
def generate_bps_basis(n_points=1024, seed=42, dtype=np.float32):
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_points, 3))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-8)
    radii = rng.random(n_points) ** (1.0 / 3.0)
    return (directions * radii[:, None]).astype(dtype)


def normalize_points(points):
    centroid = np.mean(points, axis=0, keepdims=True)
    centered = points - centroid
    max_norm = np.linalg.norm(centered, axis=1).max()
    if max_norm < 1e-6:
        max_norm = 1.0
    return (centered / max_norm).astype(np.float32)


def compute_bps_from_mesh(mesh, n_bps=1024, surface_points=4096, seed=42):
    basis = generate_bps_basis(n_bps, seed=seed)
    dense_points = mesh.sample(surface_points, return_index=False).astype(np.float32)
    norm_points = normalize_points(dense_points)
    # shape: [n_bps, n_dense, 3]
    diff = basis[:, None, :] - norm_points[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    bps = dists.min(axis=1).astype(np.float32)
    return bps, basis


# --------------------------
# Object point + affordance condition
# --------------------------
def sample_object_points_with_normals(mesh, n_points=100, seed=42):
    np.random.seed(seed)
    points, face_indices = mesh.sample(n_points, return_index=True)
    normals = mesh.face_normals[face_indices]
    return points.astype(np.float32), normals.astype(np.float32)


def load_affordance_scores(npz_bytes):
    with np.load(io.BytesIO(npz_bytes)) as data:
        if "sampled_scores" not in data:
            raise ValueError("affordance npz 需要包含 sampled_scores")
        return data["sampled_scores"].astype(np.float32)


def load_gt_contact(npz_bytes):
    with np.load(io.BytesIO(npz_bytes)) as data:
        for key in ["points", "valid"]:
            if key not in data:
                raise ValueError(f"GT contact npz 缺少字段: {key}")
        points = data["points"].astype(np.float32)
        valid = data["valid"].astype(np.float32)
    return points, valid


def coerce_obj_pose(trajectory):
    traj = np.asarray(trajectory, dtype=np.float32)
    if traj.ndim == 2:
        if traj.shape != (4, 4):
            raise ValueError(f"trajectory 2D 形状非法: {traj.shape}, 需要(4,4)")
        traj = traj[None, ...]
    elif traj.ndim == 3:
        if traj.shape[1:] != (4, 4):
            raise ValueError(f"trajectory 3D 形状非法: {traj.shape}, 需要(T,4,4)")
    else:
        raise ValueError(f"trajectory 维度非法: {traj.shape}")
    return traj


def build_condition(mesh, trajectory, affordance_scores):
    obj_pose = coerce_obj_pose(trajectory)  # [T,4,4]
    T = obj_pose.shape[0]

    n_points = int(affordance_scores.shape[0])
    sampled_points, sampled_normals = sample_object_points_with_normals(mesh, n_points=n_points, seed=42)

    bps, basis = compute_bps_from_mesh(mesh, n_bps=1024, surface_points=4096, seed=42)

    # obj_points: [T,N,7] = world_xyz + local_normal + affordance
    obj_points = np.zeros((T, n_points, 7), dtype=np.float32)
    points_h = np.ones((n_points, 4), dtype=np.float32)
    points_h[:, :3] = sampled_points

    for t in range(T):
        pose = obj_pose[t]
        world = (pose @ points_h.T).T[:, :3]
        obj_points[t] = np.concatenate([world, sampled_normals, affordance_scores[:, None]], axis=1)

    obj_bps = np.repeat(bps[None, :], T, axis=0)  # [T,1024]

    return {
        "obj_pose": obj_pose,
        "obj_points": obj_points,
        "obj_bps": obj_bps,
        "bps_basis": basis,
        "bps_vec": bps,
        "sampled_points_local": sampled_points,
        "sampled_normals_local": sampled_normals,
        "affordance_scores": affordance_scores,
    }


# --------------------------
# Visualization
# --------------------------
def _set_axes_equal(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = np.max(maxs - mins) * 0.5
    radius = max(radius, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


def _compute_axis_limits_from_points(points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = np.max(maxs - mins) * 0.5
    radius = max(radius, 1e-6)
    return {
        "center": center.astype(np.float32),
        "radius": float(radius),
    }


def _apply_axis_limits(ax, axis_limits):
    center = axis_limits["center"]
    radius = float(axis_limits["radius"])
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


def _compute_sequence_axis_limits(mesh, obj_pose, contacts, valid):
    T = min(obj_pose.shape[0], contacts.shape[0], valid.shape[0])
    mesh_verts_local = np.asarray(mesh.vertices, dtype=np.float32)
    verts_h = np.ones((mesh_verts_local.shape[0], 4), dtype=np.float32)
    verts_h[:, :3] = mesh_verts_local

    points_for_limits = []
    for t in range(T):
        pose = obj_pose[t]
        world_verts = (pose @ verts_h.T).T[:, :3]
        stride = max(1, len(world_verts) // 4000)
        points_for_limits.append(world_verts[::stride])
        c_t = contacts[t].reshape(-1, 3)
        v_t = valid[t].reshape(-1) > 0.5
        if np.any(v_t):
            points_for_limits.append(c_t[v_t])

    if not points_for_limits:
        points_for_limits = [mesh_verts_local[: min(len(mesh_verts_local), 4000)]]

    all_points = np.vstack(points_for_limits)
    return _compute_axis_limits_from_points(all_points)


def visualize_bps(mesh, bps_basis, bps_vec):
    R = np.array([[1, 0, 0],
                  [0, 0,-1],
                  [0, 1, 0]], dtype=np.float32)

    verts = np.asarray(mesh.vertices, dtype=np.float32) @ R.T
    bps_basis = np.asarray(bps_basis, dtype=np.float32) @ R.T

    fig = plt.figure(figsize=(5,5.75), facecolor="#ffffff")
    gs = fig.add_gridspec(1, 2, width_ratios=[20,1])
    ax = fig.add_subplot(gs[0], projection="3d", facecolor="#f8f9fa")
    cax = fig.add_subplot(gs[1])

    ax.scatter(
        verts[::max(1, len(verts)//4000), 0],
        verts[::max(1, len(verts)//4000), 1],
        verts[::max(1, len(verts)//4000), 2],
        c="#aaaaaa", s=1, alpha=0.3
    )

    sc = ax.scatter(
        bps_basis[:,0],
        bps_basis[:,1],
        bps_basis[:,2],
        c=bps_vec,
        cmap="plasma",
        s=10
    )

    fig.colorbar(sc, cax=cax)

    _set_axes_equal(ax, np.vstack([verts[:min(len(verts),20000)], bps_basis]))

    ax.set_title("BPS Basis + Distance", color="#222222")
    ax.set_axis_off()

    return fig


def visualize_affordance_points(points_local, affordance_scores):
    R = np.array([[1, 0, 0],
                  [0, 0,-1],
                  [0, 1, 0]], dtype=np.float32)
    points_local = np.asarray(points_local, dtype=np.float32) @ R.T
    fig = plt.figure(figsize=(6, 5), facecolor="#ffffff")
    ax = fig.add_subplot(111, projection="3d", facecolor="#f8f9fa")
    ax.scatter(points_local[:, 0], points_local[:, 1], points_local[:, 2], c="#aaaaaa", s=12, alpha=0.8,
               edgecolors="none", linewidths=0)
    hot = affordance_scores > 1e-6
    if np.any(hot):
        colors = np.zeros((hot.sum(), 4), dtype=np.float32)
        colors[:, 0] = 1.0
        colors[:, 3] = np.clip(affordance_scores[hot], 0.1, 1.0)
        ax.scatter(points_local[hot, 0], points_local[hot, 1], points_local[hot, 2], c=colors, s=30,
                   edgecolors="none", linewidths=0)
    _set_axes_equal(ax, points_local)
    ax.set_title("Object points(gray) + affordance(red)", color="#222222")
    ax.set_axis_off()
    return fig


def visualize_trajectory(obj_pose):
    traj = obj_pose[:, :3, 3]
    fig = plt.figure(figsize=(6, 5), facecolor="#ffffff")
    ax = fig.add_subplot(111, projection="3d", facecolor="#f8f9fa")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c="#2563eb", linewidth=2)
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c="#16a34a", s=40, label="start")
    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c="#dc2626", s=40, label="end")
    _set_axes_equal(ax, traj)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Object trajectory", color="#222222")
    ax.set_axis_off()
    return fig


def render_contacts_gif(
    mesh,
    obj_pose,
    contacts,
    valid,
    out_fps=12,
    axis_limits=None,
    return_axis_limits=False,
):
    T = min(obj_pose.shape[0], contacts.shape[0], valid.shape[0])
    R = np.array([[1, 0, 0],
                [0, 0,-1],
                [0, 1, 0]], dtype=np.float32)
    mesh_verts_local = np.asarray(mesh.vertices, dtype=np.float32)
    verts_h = np.ones((mesh_verts_local.shape[0], 4), dtype=np.float32)
    verts_h[:, :3] = mesh_verts_local

    # ------------------------------------------------------------------
    # Determine axis limits once for the whole sequence
    # ------------------------------------------------------------------
    if axis_limits is None:
        axis_limits = _compute_sequence_axis_limits(mesh, obj_pose, contacts, valid)
    axis_limits["center"] = axis_limits["center"] @ R.T

    frames = []
    for t in range(T):
        pose = obj_pose[t]
        world_verts = (pose @ verts_h.T).T[:, :3]
        world_verts = world_verts @ R.T

        fig = plt.figure(figsize=(6, 5), facecolor="#ffffff")
        ax = fig.add_subplot(111, projection="3d", facecolor="#f8f9fa")
        # ax.view_init(elev=10, azim=20)

        stride = max(1, len(world_verts) // 4000)
        ax.scatter(
            world_verts[::stride, 0],
            world_verts[::stride, 1],
            world_verts[::stride, 2],
            c="#aaaaaa", s=1, alpha=0.30, edgecolors="none", linewidths=0,
        )

        c_t = contacts[t].reshape(-1, 3)
        c_t = c_t @ R.T
        v_t = valid[t].reshape(-1) > 0.5

        if np.any(v_t):
            cp = c_t[v_t]
            ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2],
                       c="#ff3b30", s=45, edgecolors="#333333", linewidths=0.3)

        # Apply the shared (or newly computed) limits – no per-frame re-adaptation
        _apply_axis_limits(ax, axis_limits)
        ax.set_title(f"frame {t:03d}", color="#222222", fontsize=10)
        ax.set_axis_off()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)

    gif_io = io.BytesIO()
    imageio.mimsave(gif_io, frames, format="gif", fps=out_fps, loop=0)
    gif_io.seek(0)
    gif_bytes = gif_io.read()

    if return_axis_limits:
        return gif_bytes, axis_limits
    return gif_bytes


# --------------------------
# Model inference
# --------------------------
def load_contact_model(weight_path, device):
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch 不可用，无法加载模型")

    model = contact_point_generator(smpl=None)
    ckpt = torch.load(weight_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError("模型权重格式不支")

    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, missing, unexpected


def infer_contact(model, cond, device):
    with torch.no_grad():
        data = {
            "obj_bps": torch.from_numpy(cond["obj_bps"]).unsqueeze(0).to(device),
            "obj_pose": torch.from_numpy(cond["obj_pose"]).unsqueeze(0).to(device),
            "obj_points": torch.from_numpy(cond["obj_points"]).unsqueeze(0).to(device),
        }
        out = model(data)
        pred_points = out["pred_contact_points"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        pred_valid = torch.sigmoid(out["pred_contact_logits"]).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return pred_points, pred_valid


# --------------------------
# Flask routes
# --------------------------
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/prepare", methods=["POST"])
def prepare():
    """上传输入文件并构建模型条件 + 可视化。"""
    try:
        obj_file = request.files.get("obj_file")
        traj_file = request.files.get("traj_file")
        aff_file = request.files.get("aff_file")
        gt_file = request.files.get("gt_file")

        if not (obj_file and traj_file and aff_file and gt_file):
            return jsonify({"error": "请上传 mesh.obj、trajectory.npy、affordance.npz、gt_contact.npz"}), 400

        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tf:
            obj_path = tf.name
            obj_file.save(obj_path)

        mesh = trimesh.load(obj_path, force="mesh")
        os.unlink(obj_path)

        trajectory = np.load(io.BytesIO(traj_file.read()), allow_pickle=True)
        aff_scores = load_affordance_scores(aff_file.read())
        gt_points, gt_valid = load_gt_contact(gt_file.read())

        cond = build_condition(mesh, trajectory, aff_scores)

        fig_bps = visualize_bps(mesh, cond["bps_basis"], cond["bps_vec"])
        fig_aff = visualize_affordance_points(cond["sampled_points_local"], cond["affordance_scores"])
        fig_traj = visualize_trajectory(cond["obj_pose"])

        bps_img = fig_to_b64(fig_bps)
        aff_img = fig_to_b64(fig_aff)
        traj_img = fig_to_b64(fig_traj)
        plt.close("all")

        payload = {
            "obj_pose": bytes_to_b64(cond["obj_pose"].astype(np.float32).tobytes()),
            "obj_points": bytes_to_b64(cond["obj_points"].astype(np.float32).tobytes()),
            "obj_bps": bytes_to_b64(cond["obj_bps"].astype(np.float32).tobytes()),
            "gt_points": bytes_to_b64(gt_points.astype(np.float32).tobytes()),
            "gt_valid": bytes_to_b64(gt_valid.astype(np.float32).tobytes()),
            "mesh_obj": bytes_to_b64(trimesh.exchange.export.export_mesh(mesh, "obj").encode("utf-8")),
            "T": int(cond["obj_pose"].shape[0]),
            "N": int(cond["obj_points"].shape[1]),
            "gt_T": int(gt_points.shape[0]),
        }

        return jsonify({
            "bps_img": bps_img,
            "aff_img": aff_img,
            "traj_img": traj_img,
            "payload": payload,
            "stats": {
                "frames": int(cond["obj_pose"].shape[0]),
                "sampled_points": int(cond["obj_points"].shape[1]),
                "bps_dim": int(cond["obj_bps"].shape[-1]),
                "mesh_vertices": int(len(mesh.vertices)),
            }
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/run_model", methods=["POST"])
def run_model():
    try:
        if not TORCH_AVAILABLE:
            return jsonify({"error": "当前环境缺少torch，无法推理"}), 500

        data = request.get_json(force=True)
        weight_path = data.get("weight_path", "").strip()
        payload = data.get("payload", {})
        if not weight_path or not os.path.exists(weight_path):
            return jsonify({"error": f"模型权重不存在: {weight_path}"}), 400

        T = int(payload["T"])
        N = int(payload["N"])
        gt_T = int(payload["gt_T"])

        obj_pose = np.frombuffer(b64_to_bytes(payload["obj_pose"]), dtype=np.float32).reshape(T, 4, 4)
        obj_points = np.frombuffer(b64_to_bytes(payload["obj_points"]), dtype=np.float32).reshape(T, N, 7)
        obj_bps = np.frombuffer(b64_to_bytes(payload["obj_bps"]), dtype=np.float32).reshape(T, 1024)

        gt_points = np.frombuffer(b64_to_bytes(payload["gt_points"]), dtype=np.float32).reshape(gt_T, 2, 2, 3)
        gt_valid = np.frombuffer(b64_to_bytes(payload["gt_valid"]), dtype=np.float32).reshape(gt_T, 2, 2)

        mesh_obj = b64_to_bytes(payload["mesh_obj"]).decode("utf-8")
        mesh = trimesh.load(io.StringIO(mesh_obj), file_type="obj", force="mesh")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, missing, unexpected = load_contact_model(weight_path, device)

        pred_points, pred_valid = infer_contact(model, {
            "obj_pose": obj_pose,
            "obj_points": obj_points,
            "obj_bps": obj_bps,
        }, device)

        gt_gif, gt_axis_limits = render_contacts_gif(
            mesh, obj_pose, gt_points, gt_valid, out_fps=12, return_axis_limits=True
        )
        pred_gif = render_contacts_gif(
            mesh, obj_pose, pred_points, pred_valid, out_fps=12, axis_limits=gt_axis_limits
        )

        pred_bin = (pred_valid > 0.5).astype(np.float32)
        total_pred = int(pred_bin.sum())
        total_gt = int((gt_valid > 0.5).sum())

        return jsonify({
            "gt_gif": bytes_to_b64(gt_gif),
            "pred_gif": bytes_to_b64(pred_gif),
            "summary": {
                "pred_valid_sum": total_pred,
                "gt_valid_sum": total_gt,
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
            }
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Contact Visualizer</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 32px 24px 48px;
      background: #f0f2f5;
      color: #222;
    }

    /* ── Page wrapper keeps everything to one consistent width ── */
    .page {
      max-width: 1100px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    h1 {
      margin: 0 0 4px 0;
      font-size: 22px;
      font-weight: 700;
      color: #111;
      letter-spacing: -0.3px;
    }

    /* ── Cards ── */
    .card {
      background: #ffffff;
      border: 1px solid #e2e4e8;
      border-radius: 10px;
      padding: 20px 24px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      width: 100%;
    }

    .card-title {
      font-size: 13px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: #888;
      margin: 0 0 16px 0;
    }

    /* ── Upload grid: 2-column label+input rows ── */
    .upload-grid {
      display: grid;
      grid-template-columns: 180px 1fr;
      row-gap: 10px;
      column-gap: 16px;
      align-items: center;
    }

    .upload-label {
      font-size: 13px;
      color: #444;
      font-weight: 500;
      white-space: nowrap;
    }

    input[type='file'] {
      width: 100%;
      font-size: 13px;
      color: #333;
      background: #f8f9fb;
      border: 1px solid #d0d3d8;
      border-radius: 6px;
      padding: 6px 10px;
      cursor: pointer;
    }
    input[type='file']:hover { border-color: #aab; }

    input[type='text'] {
      width: 100%;
      font-size: 13px;
      color: #333;
      background: #f8f9fb;
      border: 1px solid #d0d3d8;
      border-radius: 6px;
      padding: 8px 12px;
      outline: none;
      transition: border-color 0.15s;
    }
    input[type='text']:focus { border-color: #3b82f6; }

    .input-row {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .input-row label { font-size: 13px; color: #444; font-weight: 500; }

    /* ── Buttons ── */
    .btn-row { margin-top: 18px; display: flex; align-items: center; gap: 16px; }

    button {
      padding: 9px 20px;
      border: none;
      border-radius: 7px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      transition: opacity 0.15s, transform 0.1s;
    }
    button:hover { opacity: 0.88; }
    button:active { transform: scale(0.97); }
    .btn-blue { background: #3b82f6; color: #fff; }
    .btn-red  { background: #ef4444; color: #fff; }

    #stats {
      font-size: 12px;
      color: #666;
      font-family: "SFMono-Regular", Consolas, monospace;
    }

    /* ── Visualization row: 3 equal cards ── */
    .viz-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      width: 100%;
    }

    .viz-card {
      background: #ffffff;
      border: 1px solid #e2e4e8;
      border-radius: 10px;
      padding: 16px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    .viz-card-title {
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: #888;
      margin: 0 0 10px 0;
    }

    .viz-card img {
      width: 100%;
      border-radius: 6px;
      display: block;
    }

    /* ── GIF row: 2 equal cards ── */
    .gif-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      width: 100%;
    }

    .gif-card {
      background: #ffffff;
      border: 1px solid #e2e4e8;
      border-radius: 10px;
      padding: 16px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      min-height: 60px;
    }

    .gif-card-title {
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: #888;
      margin: 0 0 10px 0;
    }

    .gif-card img {
      width: 100%;
      border-radius: 6px;
      display: block;
    }

    /* ── Error ── */
    pre.err {
      color: #dc2626;
      white-space: pre-wrap;
      font-size: 12px;
      margin: 0;
      font-family: "SFMono-Regular", Consolas, monospace;
    }

    /* ── Divider ── */
    .section-label {
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 1px;
      text-transform: uppercase;
      color: #aaa;
      margin: 0;
    }
  </style>
</head>
<body>
<div class="page">

  <h1>Contact Visualizer</h1>

  <!-- ── Step 1: Upload ── -->
  <div class="card">
    <div class="card-title">Step 1 — Upload Input Files</div>
    <div class="upload-grid">
      <span class="upload-label">mesh.obj</span>
      <input type="file" id="obj_file" accept=".obj">

      <span class="upload-label">trajectory.npy (T,4,4)</span>
      <input type="file" id="traj_file" accept=".npy">

      <span class="upload-label">affordance.npz</span>
      <input type="file" id="aff_file" accept=".npz">

      <span class="upload-label">gt_contact.npz</span>
      <input type="file" id="gt_file" accept=".npz">
    </div>
    <div class="btn-row">
      <button class="btn-blue" id="btn_prepare">准备条件并可视化</button>
      <span id="stats"></span>
    </div>
  </div>

  <!-- ── Visualizations ── -->
  <div class="viz-row">
    <div class="viz-card">
      <div class="viz-card-title">BPS</div>
      <div id="bps_box"></div>
    </div>
    <div class="viz-card">
      <div class="viz-card-title">Affordance 点云</div>
      <div id="aff_box"></div>
    </div>
    <div class="viz-card">
      <div class="viz-card-title">Trajectory</div>
      <div id="traj_box"></div>
    </div>
  </div>

  <!-- ── Step 2: Inference ── -->
  <div class="card">
    <div class="card-title">Step 2 — Model Inference</div>
    <div class="input-row">
      <label>模型权重路径（.pkl）</label>
      <input type="text" id="weight_path" placeholder="/path/to/your_epoch200.pkl">
    </div>
    <div class="btn-row">
      <button class="btn-red" id="btn_run">运行模型并生成 GT / Pred GIF</button>
    </div>
  </div>

  <!-- ── GIF Output ── -->
  <div class="gif-row">
    <div class="gif-card">
      <div class="gif-card-title">GT Contact GIF</div>
      <div id="gt_box"></div>
    </div>
    <div class="gif-card">
      <div class="gif-card-title">Pred Contact GIF</div>
      <div id="pred_box"></div>
    </div>
  </div>

  <pre class="err" id="err"></pre>

</div><!-- /page -->

<script>
let payloadCache = null;

function setImg(id, b64, mime='image/png') {
  document.getElementById(id).innerHTML = `<img src="data:${mime};base64,${b64}" />`;
}

async function postForm(url, fd) {
  const r = await fetch(url, { method:'POST', body:fd });
  return await r.json();
}

async function postJSON(url, obj) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(obj)
  });
  return await r.json();
}

function showErr(msg) { document.getElementById('err').textContent = msg || ''; }

document.getElementById('btn_prepare').onclick = async () => {
  showErr('');
  const fd = new FormData();
  fd.append('obj_file',  document.getElementById('obj_file').files[0]);
  fd.append('traj_file', document.getElementById('traj_file').files[0]);
  fd.append('aff_file',  document.getElementById('aff_file').files[0]);
  fd.append('gt_file',   document.getElementById('gt_file').files[0]);

  const data = await postForm('/prepare', fd);
  if (data.error) { showErr(data.error); return; }

  payloadCache = data.payload;
  setImg('bps_box',  data.bps_img);
  setImg('aff_box',  data.aff_img);
  setImg('traj_box', data.traj_img);

  const s = data.stats;
  document.getElementById('stats').textContent =
    `frames=${s.frames}  •  sampled_points=${s.sampled_points}  •  bps_dim=${s.bps_dim}  •  mesh_vertices=${s.mesh_vertices}`;
};

document.getElementById('btn_run').onclick = async () => {
  showErr('');
  if (!payloadCache) { showErr('请先点击"准备条件并可视化"'); return; }
  const weight_path = document.getElementById('weight_path').value.trim();
  if (!weight_path) { showErr('请填写模型权重路径'); return; }

  const data = await postJSON('/run_model', { weight_path, payload: payloadCache });
  if (data.error) { showErr(data.error); return; }

  setImg('gt_box',   data.gt_gif,   'image/gif');
  setImg('pred_box', data.pred_gif, 'image/gif');

  const s = data.summary;
  const el = document.getElementById('stats');
  el.textContent += `  •  pred_valid=${s.pred_valid_sum}  •  gt_valid=${s.gt_valid_sum}  •  missing_keys=${s.missing_keys}  •  unexpected_keys=${s.unexpected_keys}`;
};
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("=" * 55)
    print("Contact Visualizer 启动中")
    print("访问: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=5000)