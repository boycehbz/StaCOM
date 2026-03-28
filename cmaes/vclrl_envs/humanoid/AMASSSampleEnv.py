'''
 @FileName    : AMASSSampleEnv.py
 @EditTime    : 2021-12-20 15:00:02
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : Create an environment for the current character, scene and mocap data. 
                Input: 
                @ target pose: A (51,) numpy for controllable joint represented with axis-angle (17*3)
                @ reference pose: A (57,) numpy for 3 base translation, 3 base rotation, 51 joint rotation
                @ current state: A (114,) numpy for 3 base translation, 3 base rotation, 51 joint rotation, 3 base linear velocity, 3 base angular velocity, 51 joint angular velocity
                Output:
                @ simulated state: A (114,) numpy

                All rotations are represented by axis-angle.
'''

from vclrl_envs.env_base import EnvBase
from vclrl_envs.amass_robot import HumanoidAMASS
from vclrl_envs.data.amass_config import lwrist, rwrist
from vclrl_envs.utils.SMPL_mocap_data import MotionCaptureData
import numpy as np
import gym
import math
import os
import torch
import json

from vclrl_envs.utils.data_utils import (
    AxisAnglePose2StateVector,
    AxisAngleState2StateVector,
    AxisAngleTarPose2QuaternionTarPose,
    StateVector2AxisAngle,
)

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi

import cv2


def load_json(path):
    with open(path) as f:
        return json.load(f)


class AMASSSampleEnv(EnvBase):
    simulation_step = 1 / 240.
    control_step    = 1 / 30.
    gravity         = 9.81
    lateralFriction = 5000000000
    KeyFrameDuration = 1 / 30.

    def __init__(
        self,
        render=False,
        data=None,
        vis_motion=False,
        smpl=None,
        person_ids=(0, 1),
        object_mesh_path=None,
        object_pose_path=None,
        object_mass=0.0,
        object_stability_seconds=0.2,
        object_stability_weight=1.0,
        object_penetration_weight=500.0,
        object_penetration_margin=0.02,
    ):
        super().__init__(HumanoidAMASS, render, data)

        self.random_start = False
        self.use_data = bool(data) and os.path.isdir(data)
        self.smpl = smpl
        self.SMPL2Humanoid = [1, 4, 7, 2, 5, 8, 3, 6, 9, 12, 15, 13, 16, 18, 14, 17, 19]

        self.person_ids  = list(person_ids)
        self.num_agents  = len(self.person_ids)
        self.object_mesh_path = object_mesh_path
        self.object_pose_path = object_pose_path
        self.object_id    = None
        self.object_poses = None
        self.max_frames   = 1
        self.start_index  = 88
        self.object_frame = self.start_index
        self.object_mass  = float(object_mass)
        self.object_stability_seconds = float(object_stability_seconds)
        self.object_stability_weight  = float(object_stability_weight)
        self.object_penetration_weight = float(object_penetration_weight)
        self.object_penetration_margin = float(object_penetration_margin)
        self.contact_targets = None

        for r in self.robots:
            r.pose_base_init()

        self._load_object()

        self.observation_space = self.robots[0].observation_space
        self.action_space      = self.robots[0].action_space
        self.temp_states = np.zeros((self.observation_space.shape[0]))
        self.times = int(self.control_step / self.simulation_step)

        if self.is_render:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

        if self.use_data:
            self._mocapData = MotionCaptureData()
            from vclrl_envs.utils.data_utils import computePose
            self._mocapData.Load(data, self.KeyFrameDuration)
            self.time = 0.
            self.firstframe = True

            if vis_motion:
                import time as _time
                count = 0
                curTime = 0.1
                pids = list(self.person_ids)[:len(self.robots)]
                motion_lists = [self._mocapData._motion_data_persons[pid] for pid in pids]
                num_agents = min(len(self.robots), len(motion_lists))

                while self._p.isConnected():
                    self.setSimTime(curTime)
                    img = None
                    if self._mocapData.img_pathes is not None:
                        img = cv2.imread(self._mocapData.img_pathes[self._frame])

                    for aid in range(num_agents):
                        robot    = self.robots[aid]
                        mlist    = motion_lists[aid]
                        frameData     = mlist[self._frame]
                        frameNextData = mlist[self._frameNext]
                        KinState = computePose(
                            self._mocapData.KeyFrameDuration,
                            frameData, frameNextData,
                            self._frameFraction, self._cycleCount,
                        )
                        robot.initializePose(
                            KinState, robot._kin_model,
                            initBase=True, initializeVelocity=True,
                        )

                    for _ in range(self.times):
                        self._p.stepSimulation()

                    curTime += self.simulation_step * self.times
                    _time.sleep(self.simulation_step * self.times)
        else:
            self._mocapData = MotionCaptureData()
            self._mocapData.KeyFrameDuration = self.KeyFrameDuration
            self._mocapData.NumFrames = 2
            self._mocapData.TotalTime = self.KeyFrameDuration
            self.time = 0.
            self.firstframe = True

    def joint_projection(self, joint, extri, intri, image, viz=False):
        im = image
        intri_ = np.insert(intri, 3, values=0., axis=1)
        temp_joint = np.insert(joint, 3, values=1., axis=1).transpose((1, 0))
        out_point = np.dot(extri, temp_joint)
        dis = out_point[2]
        out_point = (np.dot(intri_, out_point) / dis)[:-1].astype(np.int32)
        out_point = out_point.transpose(1, 0)

        if viz and im is not None:
            for i in range(len(out_point)):
                im = cv2.circle(im, tuple(out_point[i]), 5, (0, 0, 255), -1)
            ratiox = 800 / int(im.shape[0])
            ratioy = 800 / int(im.shape[1])
            ratio  = ratiox if ratiox < ratioy else ratioy
            cv2.namedWindow("mesh", 0)
            cv2.resizeWindow("mesh", int(im.shape[1] * ratio), int(im.shape[0] * ratio))
            cv2.moveWindow("mesh", 0, 0)
            cv2.imshow('mesh', im / 255.)
            cv2.waitKey(1)

        return out_point, im

    def _quat_from_matrix(self, mat):
        trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
        if trace > 0:
            s  = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (mat[2, 1] - mat[1, 2]) / s
            qy = (mat[0, 2] - mat[2, 0]) / s
            qz = (mat[1, 0] - mat[0, 1]) / s
        elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            s  = math.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
            qw = (mat[2, 1] - mat[1, 2]) / s
            qx = 0.25 * s
            qy = (mat[0, 1] + mat[1, 0]) / s
            qz = (mat[0, 2] + mat[2, 0]) / s
        elif mat[1, 1] > mat[2, 2]:
            s  = math.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
            qw = (mat[0, 2] - mat[2, 0]) / s
            qx = (mat[0, 1] + mat[1, 0]) / s
            qy = 0.25 * s
            qz = (mat[1, 2] + mat[2, 1]) / s
        else:
            s  = math.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
            qw = (mat[1, 0] - mat[0, 1]) / s
            qx = (mat[0, 2] + mat[2, 0]) / s
            qy = (mat[1, 2] + mat[2, 1]) / s
            qz = 0.25 * s
        return [qx, qy, qz, qw]

    def _load_object(self):
        if self.object_mesh_path is None or self.object_pose_path is None:
            return

        self.object_poses = np.load(self.object_pose_path)
        initial_index = min(self.start_index, len(self.object_poses) - 1)
        initial_pose  = self.object_poses[initial_index]
        initial_pos   = initial_pose[:3, 3].tolist()
        initial_quat  = self._quat_from_matrix(initial_pose[:3, :3])

        visual_shape = self._p.createVisualShape(
            shapeType=self._p.GEOM_MESH,
            fileName=self.object_mesh_path,
        )
        collision_shape = self._p.createCollisionShape(
            shapeType=self._p.GEOM_MESH,
            fileName=self.object_mesh_path,
        )
        self.object_id = self._p.createMultiBody(
            baseMass=self.object_mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=initial_pos,
            baseOrientation=initial_quat,
        )

    def _update_object_pose(self, frame_index):
        if self.object_id is None or self.object_poses is None:
            return
        frame_index = self._clamp_object_frame(frame_index)
        pose = self.object_poses[frame_index]
        pos  = pose[:3, 3].tolist()
        quat = self._quat_from_matrix(pose[:3, :3])
        self._p.resetBasePositionAndOrientation(self.object_id, pos, quat)

    def _object_target_pose(self, frame_index):
        if self.object_poses is None:
            return None, None
        frame_index = self._clamp_object_frame(frame_index)
        pose = self.object_poses[frame_index]
        pos  = pose[:3, 3]
        quat = self._quat_from_matrix(pose[:3, :3])
        return pos, quat

    def computeObjectStabilityCost(self, frame_index):
        if self.object_id is None or self.object_poses is None:
            return 0.0
        if self.object_stability_seconds <= 0 or self.object_stability_weight <= 0:
            return 0.0

        target_pos, target_quat = self._object_target_pose(frame_index)
        if target_pos is None:
            return 0.0

        state_id = self._p.saveState()
        try:
            steps = max(1, int(self.object_stability_seconds / self.simulation_step))
            for _ in range(steps):
                self._p.stepSimulation()

            cur_pos, cur_quat = self._p.getBasePositionAndOrientation(self.object_id)
            pos_err  = np.linalg.norm(np.asarray(cur_pos) - np.asarray(target_pos))
            diff_quat = self._p.getDifferenceQuaternion(cur_quat, target_quat)
            _, angle  = self._p.getAxisAngleFromQuaternion(diff_quat)
            rot_err   = abs(angle)
        finally:
            self._p.restoreState(state_id)
            self._p.removeState(state_id)

        return self.object_stability_weight * float(pos_err + rot_err)

    def _clamp_object_frame(self, frame_index):
        frame_index = int(frame_index)
        if self.object_poses is None:
            return frame_index
        max_index = len(self.object_poses) - 1
        if self.max_frames > 0:
            max_index = min(max_index, self.start_index + self.max_frames - 1)
        return max(self.start_index, min(frame_index, max_index))

    def State2SMPL(self, state):
        human_pose = np.zeros((24, 3), dtype=np.float32)
        sim_trans  = state[:3]
        sim_rot    = state[3:6]
        sim_pose   = state[6:57].reshape(-1, 3)
        human_pose[self.SMPL2Humanoid] = sim_pose
        human_pose[0] = sim_rot

        pose  = torch.tensor(human_pose, dtype=torch.float32).reshape(-1, 72)
        shape = torch.tensor(self._mocapData.human_shape, dtype=torch.float32).reshape(-1, 10)
        trans = torch.tensor(sim_trans, dtype=torch.float32).reshape(-1, 3)
        verts, joints = self.smpl(shape, pose, trans)
        halpe_joints  = self.smpl.halpe_regressor @ verts
        return halpe_joints[0].detach().numpy()

    def calcCycleCount(self, simTime, cycleTime):
        phases = simTime / cycleTime
        count  = math.floor(phases)
        return count

    def getCycleTime(self):
        keyFrameDuration = self._mocapData.KeyFrameDuration
        cycleTime = keyFrameDuration * (self._mocapData.NumFrames - 1)
        return cycleTime

    def setSimTime(self, cur_time):
        cycleTime   = self.getCycleTime()
        self._cycleCount = self.calcCycleCount(cur_time, cycleTime)
        frameTime   = cur_time - self._cycleCount * cycleTime
        if frameTime < 0:
            frameTime += cycleTime

        self._frame = int(frameTime / self._mocapData.KeyFrameDuration)
        self._frameNext = self._frame + 1
        if self._frameNext >= self._mocapData.NumFrames:
            self._frameNext = self._frame

        self._frameFraction = (
            (frameTime - self._frame * self._mocapData.KeyFrameDuration)
            / self._mocapData.KeyFrameDuration
        )

    def reset(self):
        if self.is_render:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)

        self.done = False
        if self.random_start:
            self.time = np.random.random() * self._mocapData.TotalTime
        else:
            self.time = 0.

        self.setSimTime(self.time)
        self.object_frame = self.start_index
        self._update_object_pose(self.object_frame)

        states = []
        for r in self.robots:
            s = r.reset()
            states.append(s)
        states = np.stack(states, axis=0)

        if self.is_render and len(self.robots) > 0:
            self.camera.lookat(self.robots[0].body_xyz)
        if self.is_render:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

        return states

    def computeCost(self):
        wrist_contact_weight   = 80
        penetration_cost_weight = self.object_penetration_weight

        _robot_bak    = getattr(self, "robot",     None)
        _ref_state_bak = getattr(self, "ref_state", None)

        total_cost = 0.0
        wrist_contact_total = 0.0
        wrist_contact_costs = []
        wrist_contact_costs_per_wrist = []

        wrist_contact_per_agent = self.computeWristContactCost(wrist_contact_weight)
        penetration_costs       = self.computeHumanObjectPenetrationCost(penetration_cost_weight)
        penetration_cost_total  = 0.0

        for aid, robot in enumerate(self.robots):
            self.robot     = robot
            self.ref_state = self.ref_states[aid]

            wrist_costs = [float(v) for v in wrist_contact_per_agent[aid]]
            wrist_contact_costs_per_wrist.append(wrist_costs)
            wrist_contact_cost = sum(wrist_costs)
            wrist_contact_costs.append(wrist_contact_cost)
            wrist_contact_total += wrist_contact_cost

            penetration_cost = (
                float(penetration_costs[aid]) if aid < len(penetration_costs) else 0.0
            )
            penetration_cost_total += penetration_cost

            final_cost = wrist_contact_cost + penetration_cost
            total_cost += float(final_cost)

        self.robot     = _robot_bak
        self.ref_state = _ref_state_bak
        self.last_wrist_contact_cost               = wrist_contact_total
        self.last_wrist_contact_costs              = wrist_contact_costs
        self.last_wrist_contact_costs_per_wrist    = wrist_contact_costs_per_wrist
        self.last_object_penetration_cost          = penetration_cost_total

        return total_cost

    def computeWristContactCost(self, weight, search_dist=0.5):
        HAND_LENGTH = 0.10 

        if self.object_id is None and self.contact_targets is None:
            return [[0.0, 0.0] for _ in self.robots]

        use_targets = (
            self.contact_targets is not None
            and len(self.contact_targets) == len(self.robots)
        )

        per_agent_wrist_costs      = []
        per_agent_wrist_min_dists  = []
        per_agent_wrist_has_points = []

        for agent_idx, robot in enumerate(self.robots):
            wrist_costs      = []
            wrist_min_dists  = []
            wrist_has_points = []

            for wrist_idx, wrist_link in enumerate([lwrist, rwrist]):
                left_target, right_target, left_valid, right_valid = \
                    self.contact_targets[agent_idx]
                target = right_target if wrist_idx == 1 else left_target
                valid  = right_valid  if wrist_idx == 1 else left_valid

                if not valid:
                    wrist_costs.append(0.0)
                    wrist_min_dists.append(0.0)
                    wrist_has_points.append(0.0)
                    continue

                link_state = self._p.getLinkState(robot.object_id[0], wrist_link)
                wrist_pos  = np.asarray(link_state[0], dtype=np.float64)
                dist_to_cp = float(np.linalg.norm(wrist_pos - np.asarray(target)))

                cost = weight * abs(dist_to_cp - HAND_LENGTH)
                wrist_costs.append(cost)
                wrist_min_dists.append(dist_to_cp)
                wrist_has_points.append(1.0)
            per_agent_wrist_costs.append(wrist_costs)
            per_agent_wrist_min_dists.append(wrist_min_dists)
            per_agent_wrist_has_points.append(wrist_has_points)

        self.last_wrist_contact_min_dists  = per_agent_wrist_min_dists
        self.last_wrist_contact_has_points = per_agent_wrist_has_points
        return per_agent_wrist_costs

    def computeHumanObjectPenetrationCost(self, weight=None, margin=None):
        if self.object_id is None:
            return [0.0 for _ in self.robots]

        if weight is None:
            weight = self.object_penetration_weight
        if margin is None:
            margin = self.object_penetration_margin

        per_agent_penetration_costs    = []
        per_agent_min_signed_dist      = []

        for robot in self.robots:
            link_ids = list(getattr(robot, '_linkIndicesAll', []))[1:]
            if len(link_ids) == 0:
                per_agent_penetration_costs.append(0.0)
                per_agent_min_signed_dist.append(float(margin))
                continue

            penetration_sum = 0.0
            min_signed_dist = float(margin)

            for link_idx in link_ids:
                pts = self._p.getClosestPoints(
                    robot.object_id[0],
                    self.object_id,
                    margin,
                    int(link_idx),
                    -1,
                )
                if len(pts) == 0:
                    continue
                link_min_dist   = min(float(p[8]) for p in pts)
                min_signed_dist = min(min_signed_dist, link_min_dist)
                if link_min_dist < 0.0:
                    penetration_sum += -link_min_dist

            per_agent_penetration_costs.append(float(weight) * float(penetration_sum))
            per_agent_min_signed_dist.append(float(min_signed_dist))

        self.last_object_penetration_costs          = per_agent_penetration_costs
        self.last_object_penetration_min_signed_dist = per_agent_min_signed_dist
        return per_agent_penetration_costs

    def computeCOMposVel(self, uid: int):
        pb = self._p
        num_joints   = 15
        jointIndices = range(num_joints)
        link_states  = pb.getLinkStates(uid, jointIndices, computeLinkVelocity=1)
        link_pos = np.array([s[0] for s in link_states])
        link_vel = np.array([s[-2] for s in link_states])
        tot_mass = 0.
        masses   = []
        for j in jointIndices:
            mass_, *_ = pb.getDynamicsInfo(uid, j)
            masses.append(mass_)
            tot_mass += mass_
        masses  = np.asarray(masses)[:, None]
        com_pos = np.sum(masses * link_pos, axis=0) / tot_mass
        com_vel = np.sum(masses * link_vel, axis=0) / tot_mass
        return com_pos, com_vel

    def ComputeBalanceCost(self, weight):
        per_agent_cost = []
        for r in self.robots:
            error = 0.0
            num   = len(r._end_effectors)
            if num == 0:
                per_agent_cost.append(0.0)
                continue

            simLinkStates = self._p.getLinkStates(r.object_id[0], r._linkIndicesAll)
            kinLinkStates = self._p.getLinkStates(r._kin_model,   r._linkIndicesAll)
            sim_com_pos, _ = self.computeCOMposVel(r.object_id[0])
            kin_com_pos, _ = self.computeCOMposVel(r._kin_model)

            for index in r._end_effectors:
                sim_link_pos = simLinkStates[index][0]
                kin_link_pos = kinLinkStates[index][0]
                sim_rel = [sim_com_pos[0] - sim_link_pos[0], 0.0, sim_com_pos[2] - sim_link_pos[2]]
                kin_rel = [kin_com_pos[0] - kin_link_pos[0], 0.0, kin_com_pos[2] - kin_link_pos[2]]
                diff = [sim_rel[i] - kin_rel[i] for i in range(3)]
                error += math.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)

            error /= num
            per_agent_cost.append(error)

        return weight * float(np.sum(per_agent_cost))

    def computeJointCost(self, weight):
        total_error = 0.0
        for robot in self.robots:
            link_Pos_err = 0.0
            link_Rot_err = 0.0
            mLinkWeights = [1.0] * len(robot._linkIndicesAll[1:])
            num = len(robot._linkIndicesAll[1:])

            simLinkStates = self._p.getLinkStates(robot.object_id[0], robot._linkIndicesAll[1:], 1)
            kinLinkStates = self._p.getLinkStates(robot._kin_model,   robot._linkIndicesAll[1:], 1)

            for i, _ in enumerate(robot._linkIndicesAll[1:]):
                w = mLinkWeights[i]
                simLinkInfo = simLinkStates[i]
                kinLinkInfo = kinLinkStates[i]
                diffQuat = self._p.getDifferenceQuaternion(simLinkInfo[1], kinLinkInfo[1])
                axis, angle = self._p.getAxisAngleFromQuaternion(diffQuat)
                curr_rot_err = math.sqrt(angle * angle)
                curr_pos_err = np.linalg.norm(
                    np.array(simLinkInfo[0]) - np.array(kinLinkInfo[0])
                )
                link_Pos_err += 1.0 * w * curr_pos_err
                link_Rot_err += 1.0 * w * curr_rot_err

            error = (link_Pos_err + link_Rot_err) / num
            total_error += error

        return weight * total_error

    def computeSlidingCost(self, weight):
        total_error = 0.0
        for robot in self.robots:
            error = 0.0
            pts = self._p.getClosestPoints(robot.object_id[0], self.ground_ids, 10, 2, -1)
            if len(pts) > 0:
                lankle_y = pts[0][5][1]
                if lankle_y <= 0:
                    simLinkStates = self._p.getLinkState(robot.object_id[0], 2, 1)
                    for idx in range(3):
                        error += abs(simLinkStates[6][idx])
                    error += abs(simLinkStates[7][0])
                    error += abs(simLinkStates[7][2])
            pts = self._p.getClosestPoints(robot.object_id[0], self.ground_ids, 10, 5, -1)
            if len(pts) > 0:
                rankle_y = pts[0][5][1]
                if rankle_y <= 0:
                    simLinkStates = self._p.getLinkState(robot.object_id[0], 5, 1)
                    for idx in range(3):
                        error += abs(simLinkStates[6][idx])
                    error += abs(simLinkStates[7][0])
                    error += abs(simLinkStates[7][2])
            total_error += error

        total_error /= len(self.robots)
        return weight * total_error

    def computePoseCost(self, weight):
        pose_err  = 0.0
        total_num = 0
        mJointWeights = [1.0] * 20

        for k, robot in enumerate(self.robots):
            ref_state = self.ref_states[k]
            kin_vels  = ref_state[21:]
            jointIndicesControllable = robot._jointIndicesAll
            num = len(jointIndicesControllable)

            simJointStates = self._p.getJointStatesMultiDof(robot.object_id[0], robot._linkIndicesAll[1:])
            kinJointStates = self._p.getJointStatesMultiDof(robot._kin_model,   robot._linkIndicesAll[1:])

            for i, j in enumerate(jointIndicesControllable):
                w = mJointWeights[j] if j < len(mJointWeights) else 1.0
                simJointInfo = simJointStates[j]
                kinJointInfo = kinJointStates[j]

                if len(simJointInfo[0]) == 1:
                    angle = simJointInfo[0][0] - kinJointInfo[0][0]
                    curr_pose_err = math.sqrt(angle * angle)
                elif len(simJointInfo[0]) == 4:
                    diffQuat = self._p.getDifferenceQuaternion(simJointInfo[0], kinJointInfo[0])
                    _, angle = self._p.getAxisAngleFromQuaternion(diffQuat)
                    curr_pose_err = math.sqrt(angle * angle)
                else:
                    curr_pose_err = 0.0

                pose_err  += w * curr_pose_err
                total_num += 1

        error = (pose_err / total_num) if total_num > 0 else 0.0
        return weight * error

    def computeRootCost(self, weight):
        rot_error  = 0.0
        num_agents = len(self.robots)

        for k, robot in enumerate(self.robots):
            sim_base_pos, sim_base_orn = self._p.getBasePositionAndOrientation(robot.object_id[0])
            kin_base_pos, kin_base_orn = self._p.getBasePositionAndOrientation(robot._kin_model)

            diffQuat = self._p.getDifferenceQuaternion(sim_base_orn, kin_base_orn)
            _, angle = self._p.getAxisAngleFromQuaternion(diffQuat)
            rot_error += math.sqrt(angle * angle)
            rot_error += np.linalg.norm(np.array(sim_base_pos) - np.array(kin_base_pos))

        if num_agents > 0:
            rot_error /= num_agents

        return weight * rot_error

    def computeEndEffectorCost(self, weight):
        error     = 0.0
        total_num = 0
        for robot in self.robots:
            simLinkStates = self._p.getLinkStates(robot.object_id[0], robot._linkIndicesAll[1:])
            kinLinkStates = self._p.getLinkStates(robot._kin_model,   robot._linkIndicesAll[1:])
            for index in robot._end_effectors:
                sim_link_pos = np.array(simLinkStates[index][0])
                kin_link_pos = np.array(kinLinkStates[index][0])
                error    += np.linalg.norm(sim_link_pos - kin_link_pos)
                total_num += 1
        if total_num > 0:
            error /= total_num
        return weight * error

    def computeAnkleCost(self, weight):
        ankle = [5, 2]
        num   = len(ankle)
        per_agent_cost = []

        for r in self.robots:
            error = 0.0
            simLinkStates = self._p.getLinkStates(r.object_id[0], r._linkIndicesAll[1:])
            kinLinkStates = self._p.getLinkStates(r._kin_model,   r._linkIndicesAll[1:])

            for index in ankle:
                sim_link_state = simLinkStates[index]
                kin_link_state = kinLinkStates[index]
                diffQuat = self._p.getDifferenceQuaternion(sim_link_state[1], kin_link_state[1])
                _, angle = self._p.getAxisAngleFromQuaternion(diffQuat)
                error += abs(angle)
                error += np.linalg.norm(np.array(sim_link_state[0]) - np.array(kin_link_state[0]))

        error /= num
        per_agent_cost.append(error)
        return weight * float(np.sum(per_agent_cost))

    def calcOffset(self, states):
        offsets = []
        for k, robot in enumerate(self.robots):
            state = states[k].copy()
            state[1] += 5
            state_sv = AxisAngleState2StateVector(state)
            robot.initializePose(state_sv, robot._kin_model, initBase=True, initializeVelocity=True)
            lankle_y = self._p.getClosestPoints(robot._kin_model, self.ground_ids, 10, 2, -1)[0][5][1]
            rankle_y = self._p.getClosestPoints(robot._kin_model, self.ground_ids, 10, 5, -1)[0][5][1]
            min_dis  = min(lankle_y, rankle_y)
            offsets.append(min_dis)
        return np.asarray(offsets, dtype=np.float32)

    def step(self, pack):
        tar_poses, ref_states, cur_state, ext_forces, keyp_2d = pack

        num_agents = len(self.robots)
        tar_poses  = np.asarray(tar_poses)
        ref_states = np.asarray(ref_states)
        cur_state  = np.asarray(cur_state)

        W = tar_poses.shape[0]

        if ext_forces is None:
            ext_forces = np.zeros((W, num_agents, 3), dtype=np.float32)
        else:
            ext_forces = np.asarray(ext_forces)
            if ext_forces.ndim == 2 and ext_forces.shape[1] == 3:
                ext_forces = np.repeat(ext_forces[:, None, :], num_agents, axis=1)

        self.keyp_2d = keyp_2d

        sim_states_all_windows       = []
        total_reward                 = 0.0
        contact_costs                = []
        contact_costs_per_agent      = []
        contact_costs_per_wrist      = []
        contact_min_dists_per_wrist  = []
        contact_has_points_per_wrist = []
        stability_costs              = []
        penetration_costs            = []
        penetration_costs_per_agent  = []
        penetration_min_signed_dists = []

        cur_state_sv = []
        for aid in range(num_agents):
            cur_state_sv.append(AxisAngleState2StateVector(cur_state[aid]))

        for aid, robot in enumerate(self.robots):
            robot.initializePose(
                cur_state_sv[aid], robot.object_id[0],
                initBase=True, initializeVelocity=True,
            )

        for w in range(W):
            self._update_object_pose(self.object_frame + w)
            self.ref_states = []
            for aid in range(num_agents):
                ref_sv = AxisAngleState2StateVector(ref_states[w, aid])
                self.ref_states.append(ref_sv)

            tar_pose_w  = tar_poses[w]
            ext_force_w = ext_forces[w]

            tar_pose_quat = []
            for aid in range(num_agents):
                self.robots[aid].initializePose(
                    self.ref_states[aid], self.robots[aid]._kin_model,
                    initBase=True, initializeVelocity=True,
                )
                tar_pose_quat.append(AxisAngleTarPose2QuaternionTarPose(tar_pose_w[aid]))

            for _ in range(self.times):
                for aid in range(num_agents):
                    self.robots[aid].apply_action(tar_pose_quat[aid], ext_force_w[aid])
                self._p.stepSimulation()

            sim_states_this_window = []
            for aid in range(num_agents):
                s = self.robots[aid].getState()
                s = StateVector2AxisAngle(s)
                sim_states_this_window.append(s)
            sim_states_all_windows.append(sim_states_this_window)

            reward = self.computeCost()

            contact_costs.append(float(getattr(self, "last_wrist_contact_cost", 0.0)))
            contact_costs_per_agent.append(
                [float(v) for v in getattr(self, "last_wrist_contact_costs", [])]
            )
            contact_costs_per_wrist.append(
                [list(map(float, wrist)) for wrist in getattr(self, "last_wrist_contact_costs_per_wrist", [])]
            )
            contact_min_dists_per_wrist.append(
                [list(map(float, wrist)) for wrist in getattr(self, "last_wrist_contact_min_dists", [])]
            )
            contact_has_points_per_wrist.append(
                [list(map(float, wrist)) for wrist in getattr(self, "last_wrist_contact_has_points", [])]
            )
            penetration_costs.append(float(getattr(self, 'last_object_penetration_cost', 0.0)))
            penetration_costs_per_agent.append(
                [float(v) for v in getattr(self, 'last_object_penetration_costs', [])]
            )
            penetration_min_signed_dists.append(
                [float(v) for v in getattr(self, 'last_object_penetration_min_signed_dist', [])]
            )

            stability_cost = self.computeObjectStabilityCost(self.object_frame + w)
            stability_costs.append(float(stability_cost))
            reward += stability_cost
            total_reward += reward

        self.object_frame = self._clamp_object_frame(self.object_frame + W)

        sim_states_all_windows = np.asarray(sim_states_all_windows)
        info = {
            "contact_costs":                 contact_costs,
            "contact_costs_per_agent":       contact_costs_per_agent,
            "contact_costs_per_wrist":       contact_costs_per_wrist,
            "contact_min_dists_per_wrist":   contact_min_dists_per_wrist,
            "contact_has_points_per_wrist":  contact_has_points_per_wrist,
            "stability_costs":               stability_costs,
            "penetration_costs":             penetration_costs,
            "penetration_costs_per_agent":   penetration_costs_per_agent,
            "penetration_min_signed_dists":  penetration_min_signed_dists,
            "contact_cost_total":            float(np.sum(contact_costs)),
            "stability_cost_total":          float(np.sum(stability_costs)),
            "penetration_cost_total":        float(np.sum(penetration_costs)),
        }
        return sim_states_all_windows, total_reward, self.done, info