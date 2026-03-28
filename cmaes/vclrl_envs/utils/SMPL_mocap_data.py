
import json
import math
import os
import pybullet as p
import numpy as np
import pickle
from vclrl_envs.utils.data_utils import Slerp, AxisAngleState2QuaternionState, AxisAnglePose2StateVector

class MotionCaptureData(object):

  def __init__(self):
    self.Reset()
    self.SMPL2Humanoid = [1,4,7,2,5,8,3,6,9,12,15,13,16,18,14,17,19]

  def Reset(self):
    self._motion_data = []
    self._motion_data_persons = {}
    self.TotalTime = 0.
    self.KeyFrameDuration = 0.
    self.NumFrames = 0
    self.person_ids = []

  def load_pkl(self, path):
      with open(path, 'rb') as f:
          param = pickle.load(f, encoding='iso-8859-1')
      return param

  def save_pkl(self, path, result):
      """"
      save pkl file
      """
      folder = os.path.dirname(path)
      if not os.path.exists(folder):
          os.makedirs(folder, exist_ok=True)

      with open(path, 'wb') as result_file:
          pickle.dump(result, result_file, protocol=2)

  def SMPL2Quaternion(self, angle):
    angle = np.array(angle)
    q = p.getQuaternionFromAxisAngle(angle/np.linalg.norm(angle), np.linalg.norm(angle))
    return q

  def load_camera_para(self, file):
      """"
      load camera parameters
      """
      campose = []
      intra = []
      campose_ = []
      distcoef = []
      intra_ = []
      f = open(file,'r')
      for line in f:
          line = line.strip('\n')
          line = line.rstrip()
          words = line.split()
          if len(words) == 3:
              intra_.append([float(words[0]),float(words[1]),float(words[2])])
          elif len(words) == 4:
              campose_.append([float(words[0]),float(words[1]),float(words[2]),float(words[3])])
          elif len(words) == 5:
              distcoef.append(np.array(words).astype(np.float))
          else:
              pass

      index = 0
      intra_t = []
      for i in intra_:
          index+=1
          intra_t.append(i)
          if index == 3:
              index = 0
              intra.append(intra_t)
              intra_t = []

      index = 0
      campose_t = []
      for i in campose_:
          index+=1
          campose_t.append(i)
          if index == 3:
              index = 0
              campose_t.append([0.,0.,0.,1.])
              campose.append(campose_t)
              campose_t = []
      
      if len(distcoef) == 0:
          distcoef = None

      return np.array(campose), np.array(intra), distcoef


  def Load(self, path, KeyFrameDuration, data_format='auto', person_ids=None):
    self.Reset()
    self.human_shape = None
    # self.person_ids = person_ids if person_ids is not None else [0]
    self.person_ids = person_ids if person_ids is not None else [0, 1]
    self.data_path = os.path.join(path, 'params')
    self.img_path = os.path.join(path, 'images')
    if os.path.exists(self.img_path):
      self.img_pathes = [os.path.join(self.img_path, im) for im in os.listdir(self.img_path)]
      self.keyp_path = os.path.join(path, 'keypoints')
      self.keyp_pathes = [os.path.join(self.keyp_path, im) for im in os.listdir(self.keyp_path)]
      self.cam_path = os.path.join(path, 'camparams.txt')
      self.extris, self.intris, _ = self.load_camera_para(self.cam_path)
    else:
      self.img_pathes, self.keyp_pathes, self.extris, self.intris = None, None, None, None
    self.KeyFrameDuration = KeyFrameDuration

    if data_format == 'pkl' or (data_format == 'auto' and os.path.isdir(self.data_path)):
      pkl_files = []
      if os.path.isdir(self.data_path):
        pkl_files = [f for f in os.listdir(self.data_path) if f.endswith('.pkl')]
      if data_format == 'pkl' or len(pkl_files) > 0:
        self._load_from_pkl(path)
      else:
        data_format = 'npz'
    if data_format != 'pkl':
      npz_files = sorted([f for f in os.listdir(path) if f.endswith('.npz')])
      if len(npz_files) == 0:
        raise FileNotFoundError('No supported motion files found in {}'.format(path))
      if len(npz_files) > 1 and len(self.person_ids) > 1:
        data_per_person = {pid: [] for pid in self.person_ids}
        global_trans_per_person = {pid: [] for pid in self.person_ids}
        for pid, npz_name in zip(self.person_ids, npz_files):
          poses, trans, betas, global_t = self._read_npz_payload(os.path.join(path, npz_name))
          person_data, person_global = self._build_person_data(poses, trans, betas, global_t, [pid])
          data_per_person[pid] = person_data[pid]
          global_trans_per_person[pid] = person_global[pid]
          if self.human_shape is None and betas is not None:
            if isinstance(betas, np.ndarray) and betas.ndim >= 2:
              self.human_shape = betas[0]
            else:
              self.human_shape = betas
        self._motion_data_persons = data_per_person
        self.global_trans_persons = global_trans_per_person
      else:
        self._load_from_npz(os.path.join(path, npz_files[0]))

    # default outputs for single-person compatibility
    main_pid = self.person_ids[0]
    self._motion_data = self._motion_data_persons.get(main_pid, [])
    self.global_trans = self.global_trans_persons.get(main_pid, [])
    self.NumFrames = min(len(self._motion_data_persons[pid]) for pid in self.person_ids)
    self.TotalTime = self.KeyFrameDuration * (self.NumFrames - 1)

  def _load_from_pkl(self, path):
    frames = sorted(os.listdir(self.data_path))
    self.NumFrames = len(frames)
    data_per_person = {pid: [] for pid in self.person_ids}
    global_trans_per_person = {pid: [] for pid in self.person_ids}
    temp = np.array([0, 0, 0])
    for f in frames:
      d_all = self.load_pkl(os.path.join(self.data_path, f))
      for pid in self.person_ids:
        person_key = 'person%02d' % pid
        if person_key not in d_all:
          continue
        d = d_all[person_key]
        frame_data = []
        frame_data += (d['transl'] - temp).reshape(-1,).tolist()
        frame_data += d['pose'][:3].reshape(-1,).tolist()
        frame_data += d['pose'].reshape(-1, 3)[self.SMPL2Humanoid].reshape(-1,).tolist()
        data_per_person[pid].append(frame_data)
        if 'global_t' in d.keys():
          global_trans_per_person[pid].append(d['global_t'])
        else:
          global_trans_per_person[pid].append(0.0)
        if self.human_shape is None:
          self.human_shape = d['betas']
    self._motion_data_persons = data_per_person
    self.global_trans_persons = global_trans_per_person

  def _read_npz_payload(self, file_path):
    raw = np.load(file_path, allow_pickle=True)
    poses = None
    trans = None
    betas = None
    global_t = None
    raw_dict = None
    arr0_payload = None
    if raw.files == ['arr_0']:
      arr0 = raw['arr_0']
      if isinstance(arr0, np.ndarray) and arr0.shape == ():
        arr0_payload = arr0.item()
        if isinstance(arr0_payload, dict):
          raw_dict = arr0_payload
        else:
          arr0_payload = np.array(arr0_payload)
    # Common naming variants
    if 'poses' in raw:
      poses = raw['poses']
    if 'trans' in raw:
      trans = raw['trans']
    elif 'transl' in raw:
      trans = raw['transl']
    if 'betas' in raw:
      betas = raw['betas']
    if 'global_t' in raw:
      global_t = raw['global_t']
    if raw_dict is not None:
      if poses is None:
        if 'poses' in raw_dict:
          poses = raw_dict['poses']
        elif 'global_orient' in raw_dict and 'body_pose' in raw_dict:
          global_orient = np.array(raw_dict['global_orient'])
          body_pose = np.array(raw_dict['body_pose'])
          pose_len = body_pose.shape[0]
          pose_full = np.zeros((pose_len, 24, 3), dtype=body_pose.dtype)
          pose_full[:, 0, :] = global_orient.reshape(pose_len, 3)
          pose_full[:, 1:22, :] = body_pose.reshape(pose_len, 21, 3)
          poses = pose_full
      if trans is None and 'trans' in raw_dict:
        trans = raw_dict['trans']
      if trans is None and 'transl' in raw_dict:
        trans = raw_dict['transl']
      if betas is None and 'betas' in raw_dict:
        betas = raw_dict['betas']
      if global_t is None and 'global_t' in raw_dict:
        global_t = raw_dict['global_t']
    if poses is None and arr0_payload is not None:
      if arr0_payload.ndim == 3:
        if arr0_payload.shape[1] == 21:
          pose_len = arr0_payload.shape[0]
          pose_full = np.zeros((pose_len, 24, 3), dtype=arr0_payload.dtype)
          pose_full[:, 1:22, :] = arr0_payload.reshape(pose_len, 21, 3)
          poses = pose_full
        else:
          poses = arr0_payload

    if trans is None and poses is not None:
      trans = np.zeros((np.array(poses).shape[0], 3), dtype=np.array(poses).dtype)
    if poses is None or trans is None:
      raise ValueError('NPZ file {} must contain poses and trans/transl entries'.format(file_path))
    return poses, trans, betas, global_t

  def _build_person_data(self, poses, trans, betas, global_t, person_ids):
    data_per_person = {pid: [] for pid in person_ids}
    global_trans_per_person = {pid: [] for pid in person_ids}
    poses = np.array(poses)
    trans = np.array(trans)
    for pid in person_ids:
      if poses.ndim == 4:
        pose_pid = poses[:, pid]
        trans_pid = trans[:, pid] if trans.ndim == 3 else trans
      else:
        pose_pid = poses
        trans_pid = trans
      for p, t in zip(pose_pid, trans_pid):
        frame_data = []
        frame_data += np.array(t).reshape(-1,).tolist()
        frame_data += np.array(p[:1]).reshape(-1,).tolist()
        frame_data += np.array(p).reshape(-1, 3)[self.SMPL2Humanoid].reshape(-1,).tolist()
        data_per_person[pid].append(frame_data)
      if global_t is not None:
        if global_t.ndim == 2:
          global_trans_per_person[pid] = list(global_t[:, pid])
        else:
          global_trans_per_person[pid] = list(global_t)
      else:
        global_trans_per_person[pid] = [0.0 for _ in range(len(data_per_person[pid]))]
    return data_per_person, global_trans_per_person

  def _load_from_npz(self, file_path):
    poses, trans, betas, global_t = self._read_npz_payload(file_path)

    data_per_person, global_trans_per_person = self._build_person_data(poses, trans, betas, global_t, self.person_ids)
    if self.human_shape is None and betas is not None:
      if isinstance(betas, np.ndarray) and betas.ndim >= 2:
        self.human_shape = betas[0]
      else:
        self.human_shape = betas
    self._motion_data_persons = data_per_person
    self.global_trans_persons = global_trans_per_person

  def Save_sample(self, path, r, target, mean, sigma, count, global_t):
    if not os.path.isdir(self.data_path):
      return
    out_path = os.path.join(path, 'params')
    frames = os.listdir(self.data_path)
    if count >= len(frames):
      return
    f = frames[count]

    d = self.load_pkl(os.path.join(self.data_path, f))

    human_pose = np.zeros((24,3), dtype=np.float32) #d['person00']['pose'].reshape(-1, 3)
    sim_trans = r[:3]
    sim_trans[1] += global_t
    sim_rot = r[3:6]
    sim_pose = r[6:57].reshape(-1, 3)
    human_pose[self.SMPL2Humanoid] = sim_pose
    human_pose[0] = sim_rot

    target_pose = np.zeros((24,3), dtype=np.float32) #d['person00']['pose'].reshape(-1, 3)
    tar_trans = target[:3]
    tar_trans[1] += global_t
    tar_rot = target[3:6]
    tar_pose = target[6:57].reshape(-1, 3)
    target_pose[self.SMPL2Humanoid] = tar_pose
    target_pose[0] = tar_rot

    d['person00']['tar_pose'] = target_pose.reshape(-1,)
    d['person00']['tar_trans'] = tar_trans.reshape(-1,)

    d['person00']['pose'] = human_pose.reshape(-1,)
    d['person00']['global_orient'] = sim_rot.reshape(-1,)
    d['person00']['transl'] = sim_trans.reshape(-1,)
    d['person00']['sigma'] = sigma.reshape(-1,)
    d['person00']['mean'] = mean.reshape(-1,)
    self.save_pkl(os.path.join(out_path, f), d)

  def Save(self, path, results):
    if not os.path.isdir(self.data_path):
      raise FileNotFoundError('Cannot save results because original data path {} is missing'.format(self.data_path))
    out_path = os.path.join(path, 'params')
    frames = os.listdir(self.data_path)
    self.NumFrames = len(frames)
    for f, r in zip(frames, results):
      frame_data = []
      d = self.load_pkl(os.path.join(self.data_path, f))

      human_pose = d['person00']['pose'].reshape(-1, 3)
      sim_trans = r[:3]
      sim_rot = r[3:6]
      sim_pose = r[6:57].reshape(-1, 3)
      human_pose[self.SMPL2Humanoid] = sim_pose
      human_pose[0] = sim_rot

      d['person00']['pose'] = human_pose.reshape(-1,)
      d['person00']['global_orient'] = sim_rot.reshape(-1,)
      d['person00']['transl'] = sim_trans.reshape(-1,)

      self.save_pkl(os.path.join(out_path, f), d)

  def Load1(self, path, KeyFrameDuration):
    data_path = os.path.join(path, 'params')
    frames = os.listdir(data_path)
    self.NumFrames = len(frames)
    self.KeyFrameDuration = KeyFrameDuration
    self.TotalTime = self.KeyFrameDuration * (self.NumFrames - 1)
    data = []
    for f in frames:
      d = self.load_pkl(os.path.join(data_path, f))['person00']
      root_pos = d['transl'].tolist()
      root_rot = d['pose'][:3].tolist()
      root_rot = list(self.SMPL2Quaternion(root_rot))
      # root_rot = root_rot[3:] + root_rot[:3]  # transform to (w x y z), it is better to use uniform format after DDL
      pose = d['pose'].reshape(-1, 3)[self.SMPL2Humanoid]
      d = [self.KeyFrameDuration] + root_pos + root_rot
      for p in pose:
        p = list(self.SMPL2Quaternion(p))
        # p = p[:3] + p[:3]
        d += p
      data.append(d)
    # self._motion_data = {'Frames':data}
    self._motion_data = {'Frames':data}

  def has_person(self, pid):
    return pid in self._motion_data_persons and len(self._motion_data_persons[pid]) > 0

  def get_person_frame(self, pid, index):
    if self.has_person(pid):
      return self._motion_data_persons[pid][index]
    raise IndexError('Person {} not available in motion data'.format(pid))

  def get_global_t(self, pid, index):
    if hasattr(self, 'global_trans_persons') and pid in self.global_trans_persons:
      return self.global_trans_persons[pid][index]
    return 0.0

  def get_min_frames(self):
    if len(self._motion_data_persons) == 0:
      return None
    lengths = [len(v) for v in self._motion_data_persons.values() if len(v) > 0]
    return min(lengths) if len(lengths) > 0 else None

  def calcCycleCount(self, simTime, cycleTime):
    phases = simTime / cycleTime
    count = math.floor(phases)
    loop = True
    #count = (loop) ? count : cMathUtil::Clamp(count, 0, 1);
    return count

  def computeCycleOffset(self):
    firstFrame = 0
    lastFrame = self.NumFrames() - 1
    frameData = self._motion_data['Frames'][0]
    frameDataNext = self._motion_data['Frames'][lastFrame]

    basePosStart = [frameData[1], frameData[2], frameData[3]]
    basePosEnd = [frameDataNext[1], frameDataNext[2], frameDataNext[3]]
    self._cycleOffset = [
        basePosEnd[0] - basePosStart[0], basePosEnd[1] - basePosStart[1],
        basePosEnd[2] - basePosStart[2]
    ]
    return self._cycleOffset
