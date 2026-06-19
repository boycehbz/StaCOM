"""
This file contains functions that are used to perform data augmentation.
"""
import os

import torch
import numpy as np
import scipy.misc
import cv2

import constants


# Defaults used by the end-to-end contact/motion generation helpers below.
DEMO_FRAME_LENGTH = 128
DEMO_NUM_AGENTS = 2
DEMO_IMAGE_HEIGHT = 1080
DEMO_IMAGE_WIDTH = 1920


def load_trajectory(path):
    """Load an object trajectory and return it as a ``(T, 4, 4)`` array."""
    print(f"Loading object trajectory: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as data:
            key = "obj_pose" if "obj_pose" in data else list(data.keys())[0]
            if key != "obj_pose":
                print(f"Using key '{key}' from npz")
            trajectory = data[key]
    else:
        trajectory = np.load(path, allow_pickle=False)

    trajectory = trajectory.astype(np.float32)
    if trajectory.ndim == 2 and trajectory.shape[1] == 16:
        trajectory = trajectory.reshape(-1, 4, 4)
    if trajectory.ndim != 3 or trajectory.shape[1:] != (4, 4):
        raise ValueError(f"Trajectory must be (T,4,4), got {trajectory.shape}")
    print(f"Trajectory shape: {trajectory.shape}")
    return trajectory


def load_affordance(path):
    """Load the ``sampled_scores`` array from an affordance NPZ file."""
    print(f"Loading affordance: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "sampled_scores" not in data:
            raise ValueError(
                "affordance npz must contain 'sampled_scores', "
                f"found keys: {list(data.keys())}"
            )
        scores = data["sampled_scores"].astype(np.float32)
    print(f"Affordance scores shape: {scores.shape}")
    return scores


def build_contact_condition(mesh, obj_traj, affordance_scores):
    """Build object-pose, point and BPS inputs for the contact model."""
    import trimesh
    from tqdm import tqdm

    print("Building contact model condition (BPS, obj_points) ...")
    num_frames = obj_traj.shape[0]
    num_points = int(affordance_scores.shape[0])

    # Use the face indices returned for the sampled points so normals and points
    # correspond to one another.
    np.random.seed(42)
    sampled_points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    sampled_points = sampled_points.astype(np.float32)
    face_normals = mesh.face_normals[face_indices].astype(np.float32)

    rng = np.random.default_rng(42)
    directions = rng.normal(size=(1024, 3))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-8)
    radii = rng.random(1024) ** (1.0 / 3.0)
    bps_basis = (directions * radii[:, None]).astype(np.float32)

    dense_points = mesh.sample(4096, return_index=False).astype(np.float32)
    centered = dense_points - dense_points.mean(axis=0, keepdims=True)
    max_norm = np.linalg.norm(centered, axis=1).max()
    normalized = centered / (max_norm if max_norm >= 1e-6 else 1.0)
    bps_vector = np.linalg.norm(
        bps_basis[:, None, :] - normalized[None, :, :], axis=-1
    ).min(axis=1).astype(np.float32)

    obj_points = np.zeros((num_frames, num_points, 7), dtype=np.float32)
    points_h = np.ones((num_points, 4), dtype=np.float32)
    points_h[:, :3] = sampled_points
    print(f"Computing world-space object points for {num_frames} frames ...")
    for frame in tqdm(range(num_frames), desc="obj_points"):
        world_points = (obj_traj[frame] @ points_h.T).T[:, :3]
        obj_points[frame] = np.concatenate(
            [world_points, face_normals, affordance_scores[:, None]], axis=1
        )

    obj_bps = np.repeat(bps_vector[None, :], num_frames, axis=0)
    print(f"obj_points shape: {obj_points.shape}, obj_bps shape: {obj_bps.shape}")
    return {"obj_pose": obj_traj, "obj_points": obj_points, "obj_bps": obj_bps}


def run_contact_model(contact_ckpt, cond, device):
    """Run contact-point inference for a prepared object condition."""
    from model.contact_point_generator import contact_point_generator

    print(f"Loading contact_point_generator from: {contact_ckpt}")
    model = contact_point_generator(smpl=None)
    checkpoint = torch.load(contact_ckpt, map_location=device)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"contact model loaded. missing={len(missing)}, unexpected={len(unexpected)}")
    model.to(device).eval()

    print("Running contact_point_generator inference ...")
    with torch.no_grad():
        data = {
            key: torch.from_numpy(cond[key]).unsqueeze(0).to(device)
            for key in ("obj_bps", "obj_pose", "obj_points")
        }
        output = model(data)

    points = output["pred_contact_points"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    valid = torch.sigmoid(output["pred_contact_logits"]).squeeze(0).detach().cpu().numpy().astype(np.float32)
    print(f"Contact points shape: {points.shape}")
    print(f"Contact valid shape:  {valid.shape}")
    return points, valid


def run_motion_model(
    motion_ckpt, obj_traj, obj_mesh_path, contact_points, contact_valid,
    gpu_index, device, use_physics=False, frame_length=DEMO_FRAME_LENGTH,
    num_agents=DEMO_NUM_AGENTS,
):
    """Run motion generation and return data ready for rendering."""
    from modules import ModelLoader
    from utils.generated_tool import _build_batch

    print("Building motion generation batch ...")
    obj_mesh_path = os.path.abspath(obj_mesh_path)
    batch = _build_batch(
        obj_traj=obj_traj, num_agents=num_agents, frame_length=frame_length,
        contact_points=contact_points, contact_valid=contact_valid,
        device=device, obj_mesh_path=obj_mesh_path, dtype=torch.float32,
    )
    batch["obj_path"] = obj_mesh_path
    workspace = os.path.join(os.path.dirname(motion_ckpt), "demo_workspace")
    os.makedirs(workspace, exist_ok=True)
    config = {
        "mode": "test", "batchsize": 1, "worker": 0, "data_folder": "",
        "model": "interhuman_flow_BPS_prior", "use_prior": 0,
        "train_loss": "", "test_loss": "", "model_type": "smplx",
        "gpu_index": gpu_index, "output": workspace, "lr": 1e-4,
        "epoch": 0, "use_sch": 0, "pretrain": 0, "pretrain_dir": "",
        "note": "", "viz": 0, "trainset": "", "testset": "",
        "frame_length": frame_length, "amp_cmu_dir": "",
        "amp_use_pretrained": 0, "amp_discriminator_ckpt": "",
        "interact_amp_use_pretrained": 0,
        "interact_amp_discriminator_ckpt": "",
    }
    print(f"Loading interhuman_flow_BPS_prior from: {motion_ckpt}")
    loader = ModelLoader(dtype=torch.float32, device=device, out_dir=workspace, **config)
    loader.load_checkpoint(motion_ckpt)
    loader.model.use_cmaes_physics = use_physics
    print(f"Physics simulation: {'ON' if use_physics else 'OFF'}")
    loader.model.eval()
    with torch.no_grad():
        predictions = loader.model(batch)

    batch_size, num_frames, agents = [int(value) for value in batch["data_shape"]]
    pose = predictions["pred_pose"].reshape(batch_size, num_frames, agents, 72)
    betas = batch["betas"].reshape(batch_size, num_frames, agents, -1)
    translation = predictions["pred_cam_t"].reshape(batch_size, num_frames, agents, 3)
    valid = batch["valid"].reshape(batch_size, num_frames, agents)
    render_sample = {
        "pose": pose[0].detach().cpu(),
        "betas": torch.zeros_like(betas[0]).detach().cpu(),
        "gt_cam_t": translation[0].detach().cpu(),
        "valid": valid[0].detach().cpu(), "obj_path": obj_mesh_path,
        "obj_pose": batch["obj_pose"][0].detach().cpu(),
        "contact_points": batch["contact_points"][0].detach().cpu(),
        "contact_valid": batch["contact_valid"][0].detach().cpu(),
    }
    return render_sample, loader


def render_frames(
    render_sample, body_model_path, output_dir, device,
    image_size=(DEMO_IMAGE_HEIGHT, DEMO_IMAGE_WIDTH),
):
    """Render a generated motion sample and return its frame directory."""
    from utils.generated_tool import (
        AZIMUTH_DEG, DISTANCE_MARGIN, ELEVATION_DEG, FOV_DEG,
        render_gt_single_view_frames,
    )
    from utils.smpl_torch_batch import SMPLXModel

    print(f"Loading SMPLX body model from: {body_model_path}")
    smpl = SMPLXModel(device=device, model_path=body_model_path, data_type=torch.float32)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    saved_paths = render_gt_single_view_frames(
        data=render_sample, smpl=smpl, output_dir=output_dir, frame_step=1,
        image_size=image_size, fov_deg=FOV_DEG, elevation_deg=ELEVATION_DEG,
        azimuth_deg=AZIMUTH_DEG, distance_margin=DISTANCE_MARGIN,
        background_color=(255, 255, 255), render_people=True,
        render_contact_points=False, contact_point_radius=0.0,
        gallery_offset_per_frame=(310, 0), gallery_frame_indices=None,
    )
    print(f"Rendered {len(saved_paths)} frame files.")
    return frames_dir

def origin2crop(keypoints, crop_data):
    old_x = crop_data['old_x']
    old_y = crop_data['old_y']
    new_x = crop_data['new_x']
    new_y = crop_data['new_y']
    cropped_shape = crop_data['new_shape']

    keypoints[:,:,0] = keypoints[:,:,0] - old_x[:,None,0]
    keypoints[:,:,1] = keypoints[:,:,1] - old_y[:,None,0]

    keypoints[:,:,0] = keypoints[:,:,0] + new_x[:,None,0]
    keypoints[:,:,1] = keypoints[:,:,1] + new_y[:,None,0]

    keypoints[:,:,0] = keypoints[:,:,0] * constants.IMG_RES / cropped_shape[:,None,1]
    keypoints[:,:,1] = keypoints[:,:,1] * constants.IMG_RES / cropped_shape[:,None,0]

    keypoints[:,:,] = 2.*keypoints[:,:,]/constants.IMG_RES - 1.
    return keypoints

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t


def surface_projection(vertices, faces, extri, intri, image, viz=False):
    """
    @ vertices: N*3, mesh vertex
    @ faces: N*3, mesh face
    @ joint: N*3, joints
    @ extri: 4*4, camera extrinsic
    @ intri: 3*3, camera intrinsic
    @ image: RGB image
    @ viz: bool, visualization
    """
    im = image
    h = im.shape[0]
    # homogeneous
    intri_ = np.insert(intri,3,values=0.,axis=1)
    temp_v = np.insert(vertices,3,values=1.,axis=1).transpose((1,0))

    # projection
    out_point = np.dot(extri, temp_v)
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1]
    out_point = (out_point.astype(np.int32)).transpose(1,0)
    
    # color
    max = dis.max()
    min = dis.min()
    t = 255./(max-min)
    color = (255, 255, 255)
    
    # draw mesh
    for f in faces:
        point = out_point[f]
        im = cv2.polylines(im, [point], True, color, 1)

    # visualization
    if viz:
        ratiox = 800/int(im.shape[0])
        ratioy = 800/int(im.shape[1])
        if ratiox < ratioy:
            ratio = ratiox
        else:
            ratio = ratioy

        cv2.namedWindow("mesh",0)
        cv2.resizeWindow("mesh",int(im.shape[1]*ratio),int(im.shape[0]*ratio))
        cv2.moveWindow("mesh",0,0)
        cv2.imshow('mesh',im/255.)
        cv2.waitKey()

    return out_point, im

def joint_projection(joint, extri, intri, image, viz=False):
    im = image
    joint = np.insert(joint, 3, values=1., axis=1)
    joint = np.dot(extri, joint.T)[:3]
    joint = np.dot(intri, joint)
    joint[:2] = joint[:2] / (joint[2] + 1e-6)
    joint = joint[:2].T

    if viz:
        viz_joint = joint.copy().astype(np.int)
        for p in viz_joint:
            im = cv2.circle(im, tuple(p), 5, (0,0,255),-1)
        ratiox = 800/int(im.shape[0])
        ratioy = 800/int(im.shape[1])
        if ratiox < ratioy:
            ratio = ratiox
        else:
            ratio = ratioy

        cv2.namedWindow("mesh",0)
        cv2.resizeWindow("mesh",int(im.shape[1]*ratio),int(im.shape[0]*ratio))
        cv2.moveWindow("mesh",0,0)
        cv2.imshow('mesh',im/255.)
        cv2.waitKey()

    return joint, im

def vis_img(name, im):
    ratiox = 800/int(im.shape[0])
    ratioy = 800/int(im.shape[1])
    if ratiox < ratioy:
        ratio = ratiox
    else:
        ratio = ratioy

    cv2.namedWindow(name,0)
    cv2.resizeWindow(name,int(im.shape[1]*ratio),int(im.shape[0]*ratio))
    #cv2.moveWindow(name,0,0)
    if im.max() > 1:
        im = im/255.
    cv2.imshow(name,im)
    if name != 'mask':
        cv2.waitKey()

def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)+1

def cam_full2crop(full_cam, center, scale, full_img_shape, focal_length):
    """
    convert the camera parameters from the crop camera to the full camera
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, 1) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    w_2, h_2 = img_w / 2., img_h / 2.
    tx, ty, tz = full_cam[:,0], full_cam[:,1], full_cam[:,2]

    bs = 2 * focal_length / (tz + 1e-9)
    s = bs / (b + 1e-9)
    x = tx - (2 * (cx - w_2) / bs)
    y = ty - (2 * (cy - h_2) / bs)
    crop_cam = torch.stack([s, x, y], dim=-1)
    return crop_cam

def cam_crop2full(crop_cam, center, scale, full_img_shape, focal_length):
    """
    convert the camera parameters from the crop camera to the full camera
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, 1) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * crop_cam[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + crop_cam[:, 1]
    ty = (2 * (cy - h_2) / bs) + crop_cam[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam

def img_crop2origin(img, origin_img, new_shape, new_x, new_y, old_x, old_y):
    img_cropped = img.numpy().transpose((1,2,0)) * 255
    img_cropped = cv2.resize(img_cropped[:,:,::-1], (new_shape[1], new_shape[0]))
    origin_img[old_y[0]:old_y[1], old_x[0]:old_x[1]] = img_cropped[new_y[0]:new_y[1], new_x[0]:new_x[1]]

    return origin_img
    
def keyp_crop2origin(keypoints, new_shape, new_x, new_y, old_x, old_y):
    keypoints = keypoints.numpy()
    keypoints[:,:-1] = (keypoints[:,:-1] + 1) * constants.IMG_RES * 0.5
    keypoints[:,0] = keypoints[:,0] / constants.IMG_RES * new_shape[1]
    keypoints[:,1] = keypoints[:,1] / constants.IMG_RES * new_shape[0]

    keypoints[:,0] = keypoints[:,0] + old_x[0]
    keypoints[:,1] = keypoints[:,1] + old_y[0]

    keypoints[:,0] = keypoints[:,0] - new_x[0]
    keypoints[:,1] = keypoints[:,1] - new_y[0]

    return keypoints

def get_crop(img_h, img_w, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1, 
                             res[1]+1], center, scale, res, invert=1))-1
    
    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], img_w) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], img_h) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(img_w, br[0])
    old_y = max(0, ul[1]), min(img_h, br[1])

    return ul, br, new_shape, new_x, new_y, old_x, old_y

def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1, 
                             res[1]+1], center, scale, res, invert=1))-1
    
    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], 
                                                        old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.ndimage.rotate(new_img, rot, reshape=False)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = cv2.resize(new_img, tuple(res), interpolation=cv2.INTER_CUBIC) #scipy.misc.imresize(new_img, res)
    return new_img, ul, br, new_shape, new_x, new_y, old_x, old_y

def uncrop(img, center, scale, orig_shape, rot=0, is_rgb=True):
    """'Undo' the image cropping/resizing.
    This function is used when evaluating mask/part segmentation.
    """
    img = img.transpose((1,2,0))
    res = img.shape[:2]
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1,res[1]+1], center, scale, res, invert=1))-1
    # size of cropped image
    crop_shape = [br[1] - ul[1], br[0] - ul[0]]

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(orig_shape, dtype=np.uint8)
    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], orig_shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], orig_shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(orig_shape[1], br[0])
    old_y = max(0, ul[1]), min(orig_shape[0], br[1])
    img = cv2.resize(img, tuple(crop_shape), interpolation=cv2.INTER_NEAREST)
    new_img[old_y[0]:old_y[1], old_x[0]:old_x[1]] = img[new_y[0]:new_y[1], new_x[0]:new_x[1]]
    return new_img

def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R,per_rdg))
    aa = (resrot.T)[0]
    return aa

def flip_img(img):
    """Flip rgb images or masks.
    channels come last, e.g. (256,256,3).
    """
    img = np.fliplr(img)
    return img

def flip_kp(kp):
    """Flip keypoints."""
    if len(kp) == 24:
        flipped_parts = constants.J24_FLIP_PERM
    elif len(kp) == 49:
        flipped_parts = constants.J49_FLIP_PERM
    elif len(kp) == 26:
        flipped_parts = constants.J26_FLIP_PERM
    kp = kp[flipped_parts]
    kp[:,0] = - kp[:,0]
    return kp

def flip_pose(pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    flipped_parts = constants.SMPL_POSE_FLIP_PERM
    pose = pose[flipped_parts]
    # we also negate the second and the third dimension of the axis-angle
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    return pose

def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    """Increase the size of the bounding box to match the target shape."""
    if target_aspect_ratio is None:
        return input_shape

    try:
        w , h = input_shape
    except (ValueError, TypeError):
        return input_shape

    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    if h_new < h or w_new < w:
        breakpoint()
    return np.array([w_new, h_new])
