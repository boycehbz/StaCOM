import argparse
import math
import os
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import torch
import trimesh
from torch.utils.data import DataLoader, Subset

from modules import ModelLoader
from process import extract_valid, to_device
from utils.renderer_moderngl import Renderer_HOI
from utils.smpl_torch_batch import SMPLXModel

DISTANCE_MARGIN = 10.0
ELEVATION_DEG   = 40.0
FOV_DEG         = 50.0
FRAME_STEP      = 1
AZIMUTH_DEG     = -145.0


def _to_device_tensor(array, device: torch.device) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=torch.float32)
    return torch.tensor(array, device=device, dtype=torch.float32)


def _gather_valid_indices(valid_mask: Optional[torch.Tensor]) -> Sequence[int]:
    if valid_mask is None:
        return []
    if isinstance(valid_mask, torch.Tensor):
        mask = valid_mask.detach().cpu().numpy() > 0.5
    else:
        mask = np.asarray(valid_mask) > 0.5
    if mask.ndim == 2:
        mask = mask.any(axis=-1)
    return np.where(mask)[0]


def _compute_camera_parameters(
    verts: np.ndarray,
    image_size: Tuple[int, int],
    fov_deg: float,
    elevation_deg: float,
    azimuth_deg: float,
    distance_margin: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    img_h, img_w = image_size
    fov_rad = math.radians(fov_deg)
    focal = 0.5 * img_w / math.tan(fov_rad / 2)

    all_points = verts.reshape(-1, 3)
    center = all_points.mean(axis=0)

    xy_offset = all_points[:, :2] - center[:2]
    radius_xy = np.linalg.norm(xy_offset, axis=1).max()

    z_max = (all_points[:, 2] - center[2]).max()
    z_min = (all_points[:, 2] - center[2]).min()
    half_height = max(abs(z_max), abs(z_min))

    fov_y = 2 * math.atan((img_h / 2) / focal)
    dist_x = radius_xy / math.tan(fov_rad / 2) if radius_xy > 1e-6 else 0.0
    dist_y = half_height / math.tan(fov_y / 2) if half_height > 1e-6 else 0.0
    distance = max(dist_x, dist_y) + distance_margin

    elev = math.radians(elevation_deg)
    azim = math.radians(azimuth_deg)

    cam_offset = np.array([
        distance * math.cos(elev) * math.sin(azim),
        distance * math.sin(elev),
        distance * math.cos(elev) * math.cos(azim),
    ])

    camera_pos = center + cam_offset
    forward = center - camera_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(up, forward)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(up, forward)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(forward, right)
    up = up / (np.linalg.norm(up) + 1e-8)

    rotation = np.stack([right, up, forward], axis=0)
    translation = -rotation @ camera_pos

    extri = np.concatenate([rotation, translation[:, None]], axis=1)
    intri = np.array([
        [focal, 0, img_w / 2],
        [0, focal, img_h / 2],
        [0, 0, 1],
    ], dtype=np.float64)

    return intri, extri


def _compose_temporal_overlay(
    images: List[np.ndarray],
    background: Tuple[int, int, int] = (255, 255, 255),
    alpha_min: float = 0.5,
    alpha_max: float = 0.9,
    offset_per_frame: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    if len(images) == 0:
        raise ValueError("No images provided for temporal overlay")
    if len(images) == 1:
        return images[0]

    h, w = images[0].shape[:2]
    background_color = np.array(background, dtype=np.float32)

    offset_dx, offset_dy = offset_per_frame
    offsets_x = [offset_dx * idx for idx in range(len(images))]
    offsets_y = [offset_dy * idx for idx in range(len(images))]

    min_offset_x = min(0, *offsets_x)
    min_offset_y = min(0, *offsets_y)
    max_offset_x = max(0, *offsets_x)
    max_offset_y = max(0, *offsets_y)

    canvas_w = w + (max_offset_x - min_offset_x)
    canvas_h = h + (max_offset_y - min_offset_y)
    base_x = -min_offset_x
    base_y = -min_offset_y

    def create_mask(img, bg_color, threshold=30):
        diff = np.abs(img.astype(np.float32) - bg_color.reshape(1, 1, 3))
        return np.any(diff > threshold, axis=2).astype(np.float32)

    result = background_color.reshape(1, 1, 3).repeat(canvas_h, axis=0).repeat(canvas_w, axis=1)
    alpha_values = np.linspace(alpha_min, alpha_max, len(images))

    for idx, (img, alpha) in enumerate(zip(images, alpha_values)):
        img_float = img.astype(np.float32)
        mask = create_mask(img, background_color)[:, :, np.newaxis]

        x0 = int(base_x + offsets_x[idx])
        y0 = int(base_y + offsets_y[idx])
        x1 = x0 + w
        y1 = y0 + h

        region = result[y0:y1, x0:x1]
        blended = alpha * img_float + (1 - alpha) * region
        result[y0:y1, x0:x1] = mask * blended + (1 - mask) * region

    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def _extract_vertex_colors(mesh: trimesh.Trimesh, vertex_count: int) -> Optional[np.ndarray]:
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None

    vertex_colors = getattr(visual, "vertex_colors", None)
    if vertex_colors is None:
        return None

    try:
        colors = np.array(vertex_colors, dtype=np.float32, copy=True)
    except Exception:
        return None

    if colors.ndim == 1:
        colors = colors.reshape(-1, 4)
    if colors.shape[0] != vertex_count:
        return None
    if colors.shape[1] == 3:
        alpha = np.ones((vertex_count, 1), dtype=np.float32) * 255.0
        colors = np.concatenate([colors, alpha], axis=1)

    colors = colors[:, :4]
    if np.max(colors) > 1.0:
        colors /= 255.0

    rgb = colors[:, :3]
    white_mask = np.all(rgb >= 0.99, axis=1)
    uncolored_mask = np.all(rgb <= 1e-3, axis=1)
    mask = white_mask | uncolored_mask
    if np.any(mask):
        colors[mask, :3] = 0.6

    return colors


def _transform_object_vertices(
    mesh: trimesh.Trimesh,
    obj_poses: np.ndarray,
    total_frames: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    faces = np.asarray(mesh.faces, dtype=np.int32)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    vertex_colors = _extract_vertex_colors(mesh, vertices.shape[0])

    transformed = np.zeros((total_frames, vertices.shape[0], 3), dtype=np.float32)
    for frame_idx in range(min(total_frames, obj_poses.shape[0])):
        pose = obj_poses[frame_idx]
        if pose.shape != (4, 4) or not np.any(pose):
            continue
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        transformed[frame_idx] = vertices @ rotation.T + translation

    return transformed, faces, vertex_colors


def strip_near_bg(img: np.ndarray, bg: Tuple[int, int, int] = (255, 255, 255), thr: int = 30) -> np.ndarray:
    bg_arr = np.array(bg, dtype=np.float32)
    diff = np.abs(img.astype(np.float32) - bg_arr[None, None, :])
    mask = (np.any(diff > thr, axis=2)).astype(np.uint8)[:, :, None]
    return (mask * img + (1 - mask) * bg_arr[None, None, :]).astype(np.uint8)


def render_gt_single_view_frames(
    data: dict,
    smpl,
    output_dir: str,
    frame_step: int = FRAME_STEP,
    image_size: Tuple[int, int] = (1080, 1920),
    fov_deg: float = FOV_DEG,
    elevation_deg: float = ELEVATION_DEG,
    azimuth_deg: float = AZIMUTH_DEG,
    distance_margin: float = DISTANCE_MARGIN,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    render_people: bool = True,
    render_contact_points: bool = True,
    contact_point_radius: float = 0.06,
    gallery_offset_per_frame: Tuple[int, int] = (0, 0),
    gallery_frame_indices: Optional[Sequence[int]] = None,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    frame_dir = os.path.join(output_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    device = (
        next(smpl.parameters()).device
        if hasattr(smpl, "parameters") and len(list(smpl.parameters())) > 0
        else torch.device("cpu")
    )

    pose = _to_device_tensor(data["pose"], device)
    betas = _to_device_tensor(data["betas"], device)
    trans = _to_device_tensor(data["gt_cam_t"], device)

    T, num_people, _ = pose.shape
    pose_flat = pose.reshape(-1, pose.shape[-1])
    betas_flat = betas.reshape(-1, betas.shape[-1])
    trans_flat = trans.reshape(-1, trans.shape[-1])

    with torch.no_grad():
        smpl_outputs = smpl(betas_flat, pose_flat, trans_flat, halpe=True)

    joints = None
    if isinstance(smpl_outputs, tuple):
        verts = smpl_outputs[0]
        if len(smpl_outputs) > 1:
            joints = smpl_outputs[1]
    else:
        verts = smpl_outputs

    verts = verts.reshape(T, num_people, -1, 3)
    verts_np = verts.detach().cpu().numpy() if render_people else None

    joints_np = None
    if joints is not None:
        joints = joints.reshape(T, num_people, -1, 3)
        joints_np = joints.detach().cpu().numpy()

    valid_mask = data.get("valid")
    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            valid_mask_np = valid_mask.detach().cpu().numpy()
        else:
            valid_mask_np = np.asarray(valid_mask)
    else:
        valid_mask_np = None

    obj_vertices = None
    obj_faces = None
    obj_vertex_colors = None
    obj_pose = data.get("obj_pose")
    obj_path = data.get("obj_path")

    if obj_pose is not None and obj_path:
        if isinstance(obj_pose, torch.Tensor):
            obj_pose_np = obj_pose.detach().cpu().numpy()
        else:
            obj_pose_np = np.asarray(obj_pose)

        if obj_pose_np.ndim == 2:
            obj_pose_np = obj_pose_np.reshape(-1, 4, 4)

        mesh = None
        try:
            print(obj_path)
            mesh = trimesh.load(obj_path, process=False)
        except Exception:
            pass

        if mesh is None:
            try:
                print(obj_path)
                mesh = trimesh.load(obj_path, process=False)
            except Exception as exc:
                print(f"[render_gt_single_view_frames] Failed to load object mesh {obj_path}: {exc}")

        if mesh is not None:
            obj_vertices, obj_faces, obj_vertex_colors = _transform_object_vertices(mesh, obj_pose_np, T)
            if obj_vertex_colors is None:
                print(f"[render_gt_single_view_frames] Loaded {obj_path} without vertex colors")
            else:
                print(
                    f"[render_gt_single_view_frames] Loaded {obj_path} "
                    f"with vertex colors of shape {obj_vertex_colors.shape}"
                )

    contact_points_np = None
    contact_valid_np = None
    if render_contact_points:
        contact_points = data.get("contact_points")
        contact_valid = data.get("contact_valid")

        if contact_points is not None and contact_valid is not None:
            if isinstance(contact_points, torch.Tensor):
                contact_points_np = contact_points.detach().cpu().numpy()
            else:
                contact_points_np = np.asarray(contact_points)

            if contact_points_np is not None:
                contact_points_np = contact_points_np.astype(np.float32, copy=False)

            if isinstance(contact_valid, torch.Tensor):
                contact_valid_np = contact_valid.detach().cpu().numpy()
            else:
                contact_valid_np = np.asarray(contact_valid)

            if contact_valid_np is not None:
                contact_valid_np = contact_valid_np.astype(np.float32, copy=False)

            if (
                contact_points_np.ndim < 4
                or contact_valid_np.ndim < 3
                or contact_points_np.shape[0] != T
                or contact_points_np.shape[1] != contact_valid_np.shape[1]
                or contact_points_np.shape[2] != contact_valid_np.shape[2]
                or contact_points_np.shape[3] != 3
            ):
                contact_points_np = None
                contact_valid_np = None

    valid_indices = _gather_valid_indices(data.get("valid"))
    if len(valid_indices) == 0:
        valid_indices = list(range(T))

    selected = list(range(0, len(valid_indices), frame_step))
    if valid_indices[-1] not in [valid_indices[i] for i in selected]:
        selected.append(len(valid_indices) - 1)
    selected_indices = [valid_indices[i] for i in sorted(set(selected))]

    if gallery_frame_indices is not None:
        extra_frames: List[int] = []
        for frame_idx in gallery_frame_indices:
            if not isinstance(frame_idx, (int, np.integer)):
                continue
            frame_idx_int = int(frame_idx)
            if 0 <= frame_idx_int < T:
                extra_frames.append(frame_idx_int)
        if extra_frames:
            selected_indices = sorted(set(selected_indices).union(extra_frames))

    if len(selected_indices) == 0:
        selected_indices = list(valid_indices)

    img_h, img_w = image_size

    if verts_np is not None:
        verts_for_camera = verts_np.reshape(T, -1, 3)
    elif obj_vertices is not None:
        verts_for_camera = obj_vertices
    else:
        verts_for_camera = np.zeros((T, 1, 3), dtype=np.float32)

    if obj_vertices is not None and verts_np is not None:
        verts_for_camera = np.concatenate([verts_for_camera, obj_vertices], axis=1)

    camera_param_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def _get_camera_params(frame_index: int) -> Tuple[np.ndarray, np.ndarray]:
        if frame_index in camera_param_cache:
            return camera_param_cache[frame_index]

        if frame_index >= verts_for_camera.shape[0]:
            reference_idx = valid_indices[0]
            frame_points = verts_for_camera[reference_idx]
        else:
            frame_points = verts_for_camera[frame_index]

        frame_points = np.asarray(frame_points, dtype=np.float64).reshape(-1, 3)
        finite_mask = np.isfinite(frame_points).all(axis=-1)
        if not np.any(finite_mask):
            reference_idx = valid_indices[0]
            frame_points = np.asarray(verts_for_camera[reference_idx], dtype=np.float64).reshape(-1, 3)
            finite_mask = np.isfinite(frame_points).all(axis=-1)
            if not np.any(finite_mask):
                raise ValueError("No valid vertices available to estimate camera parameters.")

        frame_points = frame_points[finite_mask]
        intri_frame, extri_frame = _compute_camera_parameters(
            frame_points, (img_h, img_w), fov_deg, elevation_deg, azimuth_deg, distance_margin
        )
        camera_param_cache[frame_index] = (intri_frame, extri_frame)
        return intri_frame, extri_frame

    background = np.ones((img_h, img_w, 3), dtype=np.uint8)
    background[:] = np.array(background_color, dtype=np.uint8)

    saved_images: List[np.ndarray] = []
    frame_to_image: Dict[int, np.ndarray] = {}
    saved_paths: List[str] = []

    renderer: Optional[Renderer_HOI] = None
    sphere_vertices = None
    sphere_faces = None

    hand_joint_indices = (9, 10)
    hand_joint_colors: Tuple[Tuple[float, float, float, float], ...] = (
        (1.0, 0.85, 0.0, 1.0),
        (1.0, 0.85, 0.0, 1.0),
    )
    hand_joint_radius = float(contact_point_radius)

    try:
        for rank, frame_idx in enumerate(selected_indices):
            meshes = []

            if render_people and verts_np is not None:
                for person_idx in range(num_people):
                    if valid_mask_np is not None and frame_idx < valid_mask_np.shape[0]:
                        if person_idx >= valid_mask_np.shape[1]:
                            continue
                        if valid_mask_np[frame_idx, person_idx] < 0.5:
                            continue
                    meshes.append((verts_np[frame_idx, person_idx], smpl.faces))

            if (
                obj_vertices is not None
                and obj_faces is not None
                and frame_idx < obj_vertices.shape[0]
                and np.any(obj_vertices[frame_idx])
            ):
                meshes.append((obj_vertices[frame_idx], obj_faces))

            if len(meshes) == 0:
                continue

            intri_frame, _ = _get_camera_params(frame_idx)
            zoom_out = max(0.2, 1.0 + 0.1 * float(distance_margin))
            focal_for_render = float(intri_frame[0, 0]) / zoom_out

            if renderer is None:
                renderer = Renderer_HOI(
                    focal_length=focal_for_render,
                    center=[intri_frame[0, 2], intri_frame[1, 2]],
                    img_w=img_w,
                    img_h=img_h,
                    use_interaction_color=True,
                )

            if (
                joints_np is not None
                and frame_idx < joints_np.shape[0]
                and joints_np.shape[1] >= num_people
            ):
                frame_joints = joints_np[frame_idx]
                for person_idx in range(num_people):
                    if frame_joints.shape[0] <= person_idx:
                        break
                    if valid_mask_np is not None and frame_idx < valid_mask_np.shape[0]:
                        if person_idx >= valid_mask_np.shape[1]:
                            continue
                        if valid_mask_np[frame_idx, person_idx] < 0.5:
                            continue
                    for hand_idx, joint_idx in enumerate(hand_joint_indices):
                        if joint_idx >= frame_joints.shape[1]:
                            continue
                        joint_pos = frame_joints[person_idx, joint_idx]
                        if not np.isfinite(joint_pos).all():
                            continue
                        if sphere_vertices is None or sphere_faces is None:
                            sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
                            sphere_vertices = np.asarray(sphere_mesh.vertices, dtype=np.float32)
                            sphere_faces = np.asarray(sphere_mesh.faces, dtype=np.int32)
                        scaled_vertices = sphere_vertices * hand_joint_radius + joint_pos
                        meshes.append((scaled_vertices.astype(np.float32), sphere_faces))

            if (
                contact_points_np is not None
                and contact_valid_np is not None
                and frame_idx < contact_points_np.shape[0]
            ):
                frame_valid = contact_valid_np[frame_idx] > 0.5
                if frame_valid.any():
                    if sphere_vertices is None or sphere_faces is None:
                        sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
                        sphere_vertices = np.asarray(sphere_mesh.vertices, dtype=np.float32)
                        sphere_faces = np.asarray(sphere_mesh.faces, dtype=np.int32)
                    frame_points = contact_points_np[frame_idx][frame_valid].reshape(-1, 3)
                    for point in frame_points:
                        if not np.isfinite(point).all():
                            continue
                        scaled_vertices = sphere_vertices * float(contact_point_radius) + point
                        meshes.append((scaled_vertices.astype(np.float32), sphere_faces))

            frame_image = renderer.render_front_view(
                meshes,
                bg_img_rgb=None,
                bg_color=background_color,
            )
            frame_image = frame_image.astype(np.uint8)
            saved_images.append(frame_image)
            frame_to_image[frame_idx] = frame_image

            filename = os.path.join(frame_dir, f"frame_{frame_idx:04d}.png")
            cv2.imwrite(filename, cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR))
            saved_paths.append(filename)

    finally:
        if renderer is not None:
            renderer.delete()

    if gallery_frame_indices is not None:
        gallery_images: List[np.ndarray] = []
        for frame_idx in gallery_frame_indices:
            if not isinstance(frame_idx, (int, np.integer)):
                continue
            img = frame_to_image.get(int(frame_idx))
            if img is not None:
                gallery_images.append(img)
        if not gallery_images:
            gallery_images = saved_images
    else:
        gallery_images = saved_images

    gallery = _compose_temporal_overlay(
        gallery_images,
        background=background_color,
        offset_per_frame=gallery_offset_per_frame,
    )
    bg_tuple = tuple(int(x) for x in np.asarray(background_color).tolist())
    gallery = strip_near_bg(gallery, bg=bg_tuple, thr=30)
    gallery_path = os.path.join(output_dir, "gallery.png")
    cv2.imwrite(gallery_path, cv2.cvtColor(gallery, cv2.COLOR_RGB2BGR))

    saved_paths.append(gallery_path)
    return saved_paths


def _parse_background(color: Sequence[int]) -> Tuple[int, int, int]:
    return tuple(int(max(0, min(255, c))) for c in color[:3])


def _load_obj_traj(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path, allow_pickle=False)
        if "obj_pose" in data:
            arr = data["obj_pose"]
        else:
            key = list(data.keys())[0]
            print(f"[load_obj_traj] Using key '{key}' from {path}")
            arr = data[key]
    else:
        arr = np.load(path, allow_pickle=False)

    arr = arr.astype(np.float32)
    if arr.ndim == 2 and arr.shape[1] == 16:
        arr = arr.reshape(-1, 4, 4)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(
            f"Object trajectory must have shape (T, 4, 4) or (T, 16), got {arr.shape}"
        )
    return arr


def _load_contact_points(
    contact_points_path: Optional[str],
    contact_valid_path: Optional[str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if contact_points_path is None or contact_valid_path is None:
        return None, None

    def _load(p):
        ext = os.path.splitext(p)[1].lower()
        if ext == ".npz":
            data = np.load(p, allow_pickle=False)
            key = list(data.keys())[0]
            return data[key].astype(np.float32)
        return np.load(p, allow_pickle=False).astype(np.float32)

    cp = _load(contact_points_path)
    cv = _load(contact_valid_path)
    return cp, cv


def _compute_obj_bps(
    mesh_path: str,
    n_bps_points: int = 1024,
    n_surface_samples: int = 4096,
    seed: int = 42,
) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_bps_points, 3))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-8)
    radii = rng.random(n_bps_points) ** (1.0 / 3.0)
    basis = (directions * radii[:, None]).astype(np.float32)

    mesh = trimesh.load(mesh_path, process=False)
    dense_points = mesh.sample(n_surface_samples, return_index=False).astype(np.float32)

    centroid = dense_points.mean(axis=0, keepdims=True)
    centered = dense_points - centroid
    max_norm = np.linalg.norm(centered, axis=1).max()
    if max_norm < 1e-6:
        max_norm = 1.0
    normalized = (centered / max_norm).astype(np.float32)

    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(normalized)
    distances, _ = nbrs.kneighbors(basis)
    return distances.squeeze(-1).astype(np.float32)


def _build_batch(
    obj_traj: np.ndarray,
    num_agents: int,
    frame_length: int,
    contact_points: Optional[np.ndarray],
    contact_valid: Optional[np.ndarray],
    device: torch.device,
    obj_mesh_path: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    T_raw = obj_traj.shape[0]
    T = frame_length if frame_length is not None else T_raw

    if T_raw >= T:
        traj = obj_traj[:T]
    else:
        pad = np.zeros((T - T_raw, 4, 4), dtype=np.float32)
        traj = np.concatenate([obj_traj, pad], axis=0)

    INPUT_FEATS = 144 + 10 + 3 + 26 * 3

    if obj_mesh_path is not None:
        print(f"[generated_tool] Computing BPS descriptor for {obj_mesh_path} ...")
        bps_descriptor = _compute_obj_bps(obj_mesh_path)
        obj_bps = np.tile(bps_descriptor[None, None, :], (1, T, 1)).astype(np.float32)
        print(f"[generated_tool] BPS descriptor computed, shape: {obj_bps.shape}")
    else:
        print("[generated_tool] No mesh path for BPS — using zero BPS features.")
        obj_bps = np.zeros((1, T, 1024), dtype=np.float32)

    if contact_points is not None and contact_valid is not None:
        cp = torch.tensor(contact_points[:T], dtype=dtype).unsqueeze(0)
        cv = torch.tensor(contact_valid[:T],  dtype=dtype).unsqueeze(0)
    else:
        cp = torch.zeros(1, T, 2, 2, 3, dtype=dtype)
        cv = torch.zeros(1, T, 2, 2,    dtype=dtype)

    batch: dict = {
        "data_shape":      torch.tensor([1, T, num_agents], dtype=torch.long),
        "x":               torch.zeros(1, T, num_agents, INPUT_FEATS, dtype=dtype),
        "pose":            torch.zeros(1, T, num_agents, 72, dtype=dtype),
        "obj_pose":        torch.tensor(traj, dtype=dtype).unsqueeze(0),
        "obj_bps":         torch.tensor(obj_bps, dtype=dtype),
        "valid":           torch.ones(1, T, num_agents, dtype=dtype),
        "betas":           torch.zeros(1, T, num_agents, 10, dtype=dtype),
        "contact_points":  cp,
        "contact_valid":   cv,
        "contact_normals": torch.zeros(1, T, 2, 2, 3, dtype=dtype),
    }

    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return batch


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render generated HOI motions from a trained model (file-based input, no dataset loader).",
    )

    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--model",       required=True)

    parser.add_argument("--obj-traj",  required=True)
    parser.add_argument("--obj-mesh",  required=True)

    parser.add_argument("--contact-points", default=None)
    parser.add_argument("--contact-valid",  default=None)

    parser.add_argument("--num-agents",   type=int, default=2)
    parser.add_argument("--frame-length", type=int, default=None)
    parser.add_argument("--use-prior",    type=int, default=0)
    parser.add_argument("--train-loss",   default="")
    parser.add_argument("--test-loss",    default="")

    parser.add_argument("--body-model",  default=None)
    parser.add_argument("--data-folder", default="data")

    parser.add_argument("--render-contact-points",    dest="render_contact_points", action="store_true",  default=True)
    parser.add_argument("--no-render-contact-points", dest="render_contact_points", action="store_false")
    parser.add_argument("--contact-radius",    type=float, default=0.00)
    parser.add_argument("--gallery-offset",    type=int, nargs=2, default=(310, 0), metavar=("DX", "DY"))
    parser.add_argument("--gallery-frames",    type=int, nargs="*", default=(0, 26, 66, 95))
    parser.add_argument("--image-height",      type=int, default=1080)
    parser.add_argument("--image-width",       type=int, default=1920)
    parser.add_argument("--background-color",  type=int, nargs=3, default=(255, 255, 255), metavar=("R", "G", "B"))

    parser.add_argument("--device",      default=None)
    parser.add_argument("--force-cpu",   action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed",        type=int, default=20)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.force_cpu:
        device = torch.device("cpu")
    elif args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gpu_index: Optional[int] = None
    if device.type == "cuda":
        gpu_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(gpu_index)

    model_workspace = os.path.abspath(os.path.join(args.output_dir, "model_workspace"))
    os.makedirs(model_workspace, exist_ok=True)
    data_folder = args.data_folder
    dtype = torch.float32

    print(f"[generated_tool] Loading object trajectory: {args.obj_traj}")
    obj_traj = _load_obj_traj(args.obj_traj)
    print(f"[generated_tool] Object trajectory shape: {obj_traj.shape}  (T={obj_traj.shape[0]} frames)")

    if args.contact_points is not None and args.contact_valid is None:
        parser.error("--contact-valid is required when --contact-points is provided.")
    if args.contact_valid is not None and args.contact_points is None:
        parser.error("--contact-points is required when --contact-valid is provided.")

    contact_points_arr, contact_valid_arr = _load_contact_points(
        args.contact_points, args.contact_valid
    )
    if contact_points_arr is not None:
        print(
            f"[generated_tool] Contact points loaded: {contact_points_arr.shape} | "
            f"valid mask: {contact_valid_arr.shape}"
        )
    else:
        print("[generated_tool] No contact points provided — skipping contact rendering.")

    batch = _build_batch(
        obj_traj=obj_traj,
        num_agents=args.num_agents,
        frame_length=args.frame_length,
        contact_points=contact_points_arr,
        contact_valid=contact_valid_arr,
        device=device,
        obj_mesh_path=os.path.abspath(args.obj_mesh),
        dtype=dtype,
    )

    config_args: dict = {
        "mode":        "test",
        "batchsize":   1,
        "worker":      args.num_workers,
        "data_folder": data_folder,
        "model":       args.model,
        "use_prior":   args.use_prior,
        "train_loss":  args.train_loss,
        "test_loss":   args.test_loss,
        "model_type":  "smplx",
        "gpu_index":   gpu_index if gpu_index is not None else 0,
        "output":      model_workspace,
        "lr":          1e-4,
        "epoch":       0,
        "use_sch":     0,
        "pretrain":    0,
        "pretrain_dir": "",
        "note":        "",
        "viz":         0,
        "trainset":    "",
        "testset":     "",
        "amp_cmu_dir":                    "",
        "amp_use_pretrained":             0,
        "amp_discriminator_ckpt":         "",
        "interact_amp_use_pretrained":    0,
        "interact_amp_discriminator_ckpt": "",
    }

    if args.frame_length is not None:
        config_args["frame_length"] = args.frame_length

    body_model_path = args.body_model or os.path.join(data_folder, "SMPLX_NEUTRAL.pkl")
    render_body_model = SMPLXModel(device=device, model_path=body_model_path, data_type=dtype)

    model_loader = ModelLoader(dtype=dtype, device=device, out_dir=model_workspace, **config_args)
    model_loader.load_checkpoint(args.checkpoint)
    model_loader.model.eval()

    with torch.no_grad():
        predictions = model_loader.model(batch)

    data_shape = batch["data_shape"]
    batch_size, frame_length, num_agents = [int(v) for v in data_shape]

    pred_pose  = predictions["pred_pose"].reshape(batch_size, frame_length, num_agents, 72)
    pred_shape = batch["betas"].reshape(batch_size, frame_length, num_agents, -1)
    pred_trans = predictions["pred_cam_t"].reshape(batch_size, frame_length, num_agents, 3)

    valid_mask     = batch["valid"].reshape(batch_size, frame_length, num_agents)
    obj_pose_batch = batch.get("obj_pose")
    contact_points = batch.get("contact_points")
    contact_valid  = batch.get("contact_valid")

    sample_pose  = pred_pose[0]
    sample_shape = torch.zeros_like(pred_shape[0])
    sample_trans = pred_trans[0]

    render_sample: dict = {
        "pose":     sample_pose.detach().cpu(),
        "betas":    sample_shape.detach().cpu(),
        "gt_cam_t": sample_trans.detach().cpu(),
        "valid":    valid_mask[0].detach().cpu(),
        "obj_path": os.path.abspath(args.obj_mesh),
    }
    if obj_pose_batch is not None:
        render_sample["obj_pose"] = obj_pose_batch[0].detach().cpu()
    if contact_points is not None:
        render_sample["contact_points"] = contact_points[0].detach().cpu()
    if contact_valid is not None:
        render_sample["contact_valid"] = contact_valid[0].detach().cpu()

    gallery_frames: Optional[Sequence[int]] = args.gallery_frames
    if gallery_frames is not None and len(gallery_frames) == 0:
        gallery_frames = None

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    background_color = _parse_background(args.background_color)

    render_gt_single_view_frames(
        data=render_sample,
        smpl=render_body_model,
        output_dir=output_dir,
        frame_step=FRAME_STEP,
        image_size=(args.image_height, args.image_width),
        fov_deg=FOV_DEG,
        elevation_deg=ELEVATION_DEG,
        azimuth_deg=AZIMUTH_DEG,
        distance_margin=DISTANCE_MARGIN,
        background_color=background_color,
        render_people=True,
        render_contact_points=args.render_contact_points,
        contact_point_radius=args.contact_radius,
        gallery_offset_per_frame=tuple(args.gallery_offset),
        gallery_frame_indices=gallery_frames,
    )

    print(f"[generated_tool] Done -> {output_dir}  (checkpoint: {args.checkpoint})")


if __name__ == "__main__":
    main()