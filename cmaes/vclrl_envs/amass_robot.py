'''
 @FileName    : robot.py
 @EditTime    : 2021-09-15 16:34:30
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import numpy as np
import gym
import os
import math
from vclrl_envs.data.amass_config import *
from vclrl_envs.utils.data_utils import StateList2StateVector, StateVector

class HumanoidAMASS():

    foot_names = ["rankle", "lankle"]

    def __init__(self, bc, sim_stpe):
        self._p = bc
        
        self.action_dim = 68
        high = np.inf * np.ones(self.action_dim)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # globals + angles + speeds + contacts
        self.state_dim = 132
        high = np.inf * np.ones(self.state_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self._jointDofCounts = [4] * 17

        self.timestep = sim_stpe

        self.P = P
        self.D = D
        self.maxForces = maxForces

        self._linkIndicesAll = [root,
                    lhip, lknee, lankle, 
                    rhip, rknee, rankle, 
                    lowerback, upperback, chest, lowerneck, upperneck,
                    lclavicle, lshoulder, lelbow, lwrist,
                    rclavicle, rshoulder, relbow, rwrist
        ]
        self._jointIndicesAll = [
                    lhip, lknee, lankle, 
                    rhip, rknee, rankle, 
                    lowerback, upperback, chest, lowerneck, upperneck,
                    lclavicle, lshoulder, lelbow,
                    rclavicle, rshoulder, relbow
        ]
        self._end_effectors = [rankle, rwrist, lankle, lwrist, upperneck]  #ankle and wrist, both left and right
        self.feet_jointsind = [rankle, lankle]
        self.feet_xyz = np.zeros((len(self.foot_names), 3))

        self.initial_y = None

    def load_robot_model(self, character_path, offset_path, lateralFriction):
        flags = (
            self._p.URDF_MAINTAIN_LINK_ORDER |
            self._p.URDF_USE_SELF_COLLISION
            | self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        )
        if os.path.exists(character_path):
            model_path = character_path
            offset = open(offset_path, 'r')
            lines = offset.readlines()
            offset = []
            for l in lines:
                offset.append(l.rstrip('\n'))
            self.character_offset = np.array(offset, np.float32).reshape(-1,)
        else:
            model_path = os.path.join("cmaes", "vclrl_envs", "data", "amass.urdf")
            self.character_offset = np.zeros((3,))

        self.base_position = (0, 0.98, 0)
        self.base_orientation = (0, 0, 0, 1)
        print(model_path)
        self.object_id = (self._p.loadURDF(model_path, 
                                            self.base_position,
                                            self.base_orientation,
                                            globalScaling=1,
                                            useFixedBase=False,
                                            flags=flags),)

        self._kin_model = self._p.loadURDF(
            model_path, [0, 0.9799, 0],
            globalScaling=1,
            useFixedBase=True,
            flags=self._p.URDF_MAINTAIN_LINK_ORDER)

        # Change Dynamics
        self._p.changeDynamics(self.object_id[0], -1, lateralFriction=lateralFriction)
        for j in range(self._p.getNumJoints(self.object_id[0])):
            self._p.changeDynamics(self.object_id[0], j, lateralFriction=lateralFriction)

        self._p.changeDynamics(self.object_id[0], -1, linearDamping=0, angularDamping=0)
        self._p.changeDynamics(self._kin_model, -1, linearDamping=0, angularDamping=0)

        self._p.changeDynamics(self.object_id[0], 2, spinningFriction=200000, rollingFriction=200000, contactStiffness=200000, contactDamping=20000)
        self._p.changeDynamics(self.object_id[0], 5, spinningFriction=200000, rollingFriction=200000, contactStiffness=200000, contactDamping=20000)

        #todo: add feature to disable simulation for a particular object. Until then, disable all collisions
        self._p.setCollisionFilterGroupMask(self._kin_model,
                                                        -1,
                                                        collisionFilterGroup=0,
                                                        collisionFilterMask=0)
        self._p.changeDynamics(
            self._kin_model,
            -1,
            activationState=self._p.ACTIVATION_STATE_SLEEP +
            self._p.ACTIVATION_STATE_ENABLE_SLEEPING +
            self._p.ACTIVATION_STATE_DISABLE_WAKEUP)
        alpha = 0.4
        self._p.changeVisualShape(self._kin_model, -1, rgbaColor=[1, 1, 1, alpha])
        for j in range(self._p.getNumJoints(self._kin_model)):
            self._p.setCollisionFilterGroupMask(self._kin_model,
                                                                j,
                                                                collisionFilterGroup=0,
                                                                collisionFilterMask=0)
            self._p.changeDynamics(
                self._kin_model,
                j,
                activationState=self._p.ACTIVATION_STATE_SLEEP +
                self._p.ACTIVATION_STATE_ENABLE_SLEEPING +
                self._p.ACTIVATION_STATE_DISABLE_WAKEUP)
            self._p.changeVisualShape(self._kin_model, j, rgbaColor=[1, 1, 1, alpha])

        for j in self._jointIndicesAll:
            #self._pybullet_client.setJointMotorControlMultiDof(self._sim_model, j, self._pybullet_client.POSITION_CONTROL, force=[1,1,1])
            self._p.setJointMotorControl2(self.object_id[0],
                                                        j,
                                                        self._p.POSITION_CONTROL,
                                                        targetPosition=0,
                                                        positionGain=0,
                                                        targetVelocity=0,
                                                        force=jointFrictionForce)
            self._p.setJointMotorControlMultiDof(
                self.object_id[0],
                j,
                self._p.POSITION_CONTROL,
                targetPosition=[0, 0, 0, 1],
                targetVelocity=[0, 0, 0],
                positionGain=0,
                velocityGain=1,
                force=[jointFrictionForce, jointFrictionForce, jointFrictionForce])
            self._p.setJointMotorControl2(self._kin_model,
                                                        j,
                                                        self._p.POSITION_CONTROL,
                                                        targetPosition=0,
                                                        positionGain=0,
                                                        targetVelocity=0,
                                                        force=0)
            self._p.setJointMotorControlMultiDof(
                self._kin_model,
                j,
                self._p.POSITION_CONTROL,
                targetPosition=[0, 0, 0, 1],
                targetVelocity=[0, 0, 0],
                positionGain=0,
                velocityGain=1,
                force=[jointFrictionForce, jointFrictionForce, 0])

        self.pose_base_init()
        
    def pose_base_init(self):
        # Initial dofs
        init_state = self.init(basePos=(0, 0.98, 0))
        # T-pose
        init_action = np.zeros(self.action_dim)
        self.base_joint_angles = self.ConvertFromAction(init_action)

    def init(self,
                # Position
                basePos=[0, 0, 0],
                baseOrn=[0, 0, 0, 1],

                leftHipRot=[0, 0, 0, 1],
                leftKneeRot=[0, 0, 0, 1],
                leftAnkleRot=[0, 0, 0, 1],
                rightHipRot=[0, 0, 0, 1],
                rightKneeRot=[0, 0, 0, 1],
                rightAnkleRot=[0, 0, 0, 1],

                lowerBackRot=[0, 0, 0, 1],
                upperBackRot=[0, 0, 0, 1],
                chestRot=[0, 0, 0, 1],
                lowerNeckRot=[0, 0, 0, 1],
                upperNeckRot=[0, 0, 0, 1],

                leftClavicleRot=[0, 0, 0, 1],
                leftShoulderRot=[0, 0, 0, 1],
                leftElbowRot=[0, 0, 0, 1],
                rightClavicleRot=[0, 0, 0, 1],
                rightShoulderRot=[0, 0, 0, 1],
                rightElbowRot=[0, 0, 0, 1],

                # Velocity
                baseLinVel=[0, 0, 0],
                baseAngVel=[0, 0, 0],

                leftHipVel=[0, 0, 0],
                leftKneeVel=[0, 0, 0],
                leftAnkleVel=[0, 0, 0],
                rightHipVel=[0, 0, 0],
                rightKneeVel=[0, 0, 0],
                rightAnkleVel=[0, 0, 0],

                lowerBackVel=[0, 0, 0],
                upperBackVel=[0, 0, 0],
                chestVel=[0, 0, 0],
                lowerNeckVel=[0, 0, 0],
                upperNeckVel=[0, 0, 0],

                leftClavicleVel=[0, 0, 0],
                leftShoulderVel=[0, 0, 0],
                leftElbowVel=[0, 0, 0],
                rightClavicleVel=[0, 0, 0],
                rightShoulderVel=[0, 0, 0],
                rightElbowVel=[0, 0, 0],
                ):

        # Position
        self._basePos=basePos
        self._baseOrn=baseOrn

        self._leftHipRot=leftHipRot
        self._leftKneeRot=leftKneeRot
        self._leftAnkleRot=leftAnkleRot
        self._rightHipRot=rightHipRot
        self._rightKneeRot=rightKneeRot
        self._rightAnkleRot=rightAnkleRot

        self._lowerBackRot=lowerBackRot
        self._upperBackRot=upperBackRot
        self._chestRot=chestRot
        self._lowerNeckRot=lowerNeckRot
        self._upperNeckRot=upperNeckRot

        self._leftClavicleRot=leftClavicleRot
        self._leftShoulderRot=leftShoulderRot
        self._leftElbowRot=leftElbowRot
        self._rightClavicleRot=rightClavicleRot
        self._rightShoulderRot=rightShoulderRot
        self._rightElbowRot=rightElbowRot

        # Velocity
        self._baseLinVel=baseLinVel
        self._baseAngVel=baseAngVel

        self._leftHipVel=leftHipVel
        self._leftKneeVel=leftKneeVel
        self._leftAnkleVel=leftAnkleVel
        self._rightHipVel=rightHipVel
        self._rightKneeVel=rightKneeVel
        self._rightAnkleVel=rightAnkleVel

        self._lowerBackVel=lowerBackVel
        self._upperBackVel=upperBackVel
        self._chestVel=chestVel
        self._lowerNeckVel=lowerNeckVel
        self._upperNeckVel=upperNeckVel

        self._leftClavicleVel=leftClavicleVel
        self._leftShoulderVel=leftShoulderVel
        self._leftElbowVel=leftElbowVel
        self._rightClavicleVel=rightClavicleVel
        self._rightShoulderVel=rightShoulderVel
        self._rightElbowVel=rightElbowVel

    def ConvertFromAction(self, action):
        # 68 dim action
        #turn action into pose

        index = 0
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        self._leftHipRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        index += 4
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        self._leftKneeRot = self._p.getQuaternionFromAxisAngle(axis, angle)
        
        index += 4
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        self._leftAnkleRot = self._p.getQuaternionFromAxisAngle(axis, angle)
        
        index += 4
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        self._rightHipRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        index += 4
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        self._rightKneeRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        index += 4
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        self._rightAnkleRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        index += 4
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        self._lowerBackRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        index += 4
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        self._upperBackRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        index += 4
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        self._chestRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        index += 4
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        self._lowerNeckRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        index += 4
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        self._upperNeckRot = self._p.getQuaternionFromAxisAngle(axis, angle)
        
        index += 4
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        self._leftClavicleRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        index += 4
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        self._leftShoulderRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        index += 4
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        self._leftElbowRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        index += 4
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        self._rightClavicleRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        index += 4
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        self._rightShoulderRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        index += 4
        angle = action[index]
        axis = [action[index + 1], action[index + 2], action[index + 3]]
        self._rightElbowRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        pose = self.GetPose()
        return pose


    def computeCycleOffset(self, _mocap_data):
        firstFrame = 0
        lastFrame = _mocap_data.NumFrames - 1
        frameData = _mocap_data._motion_data['Frames'][0]
        frameDataNext = _mocap_data._motion_data['Frames'][lastFrame]

        basePosStart = [frameData[1], frameData[2], frameData[3]]
        basePosEnd = [frameDataNext[1], frameDataNext[2], frameDataNext[3]]
        self._cycleOffset = [
            basePosEnd[0] - basePosStart[0], basePosEnd[1] - basePosStart[1],
            basePosEnd[2] - basePosStart[2]
        ]
        return self._cycleOffset


    def computePose(self, _mocap_data, _frame, _frameNext, frameFraction, _cycleCount):
        frameData = _mocap_data._motion_data[_frame]
        frameDataNext = _mocap_data._motion_data[_frameNext]
        
        frameData = AxisAnglePose2QuaternionPose(frameData, _mocap_data.KeyFrameDuration)
        frameDataNext = AxisAnglePose2QuaternionPose(frameDataNext, _mocap_data.KeyFrameDuration)

        self.Slerp(frameFraction, frameData, frameDataNext, self._p)
        #print("self._poseInterpolator.Slerp(", frameFraction,")=", pose)
        # self.computeCycleOffset(_mocap_data)
        # oldPos = self._basePos
        # self._basePos = [
        #     oldPos[0] + _cycleCount * self._cycleOffset[0],
        #     oldPos[1] + _cycleCount * self._cycleOffset[1],
        #     oldPos[2] + _cycleCount * self._cycleOffset[2]
        # ]
        pose = self.GetPose()

        return pose

        # self.Slerp(frameFraction, frameData, frameDataNext, self._p)

        # pose = self.GetPose()

        # return pose

    def Slerp(self, frameFraction, frameData, frameDataNext, bullet_client):
        keyFrameDuration = frameData[0]

        self.init()

        ##### Base Position
        basePos1Start = [frameData[1], frameData[2], frameData[3]]
        basePos1End = [frameDataNext[1], frameDataNext[2], frameDataNext[3]]
        self._basePos = [
            basePos1Start[0] + frameFraction * (basePos1End[0] - basePos1Start[0]),
            basePos1Start[1] + frameFraction * (basePos1End[1] - basePos1Start[1]),
            basePos1Start[2] + frameFraction * (basePos1End[2] - basePos1Start[2])
        ]
        self._baseLinVel = self.ComputeLinVel(basePos1Start, basePos1End, keyFrameDuration)
        
        ##### Base Orientation
        baseOrn1Start = [frameData[4], frameData[5], frameData[6], frameData[7]]
        baseOrn1Next = [frameDataNext[4], frameDataNext[5], frameDataNext[6], frameDataNext[7]]
        self._baseOrn = bullet_client.getQuaternionSlerp(baseOrn1Start, baseOrn1Next, frameFraction)
        self._baseAngVel = self.ComputeAngVel(baseOrn1Start, baseOrn1Next, keyFrameDuration, bullet_client)

        ##### Left Hip
        leftHipRotStart = [frameData[8], frameData[9], frameData[10], frameData[11]]
        leftHipRotEnd = [frameDataNext[8], frameDataNext[9], frameDataNext[10], frameDataNext[11]]
        self._leftHipRot = bullet_client.getQuaternionSlerp(leftHipRotStart, leftHipRotEnd, frameFraction)
        self._leftHipVel = self.ComputeAngVelRel(leftHipRotStart, leftHipRotEnd, keyFrameDuration, bullet_client)

        ##### Left Knee
        leftKneeRotStart = [frameData[12], frameData[13], frameData[14], frameData[15]]
        leftKneeRotEnd = [frameDataNext[12], frameDataNext[13], frameDataNext[14], frameDataNext[15]]
        self._leftKneeRot = bullet_client.getQuaternionSlerp(leftKneeRotStart, leftKneeRotEnd, frameFraction)
        self._leftKneeVel = self.ComputeAngVelRel(leftKneeRotStart, leftKneeRotEnd, keyFrameDuration, bullet_client)

        ##### Left Ankle
        leftAnkleRotStart = [frameData[16], frameData[17], frameData[18], frameData[19]]
        leftAnkleRotEnd = [frameDataNext[16], frameDataNext[17], frameDataNext[18], frameDataNext[19]]
        self._leftAnkleRot = bullet_client.getQuaternionSlerp(leftAnkleRotStart, leftAnkleRotEnd, frameFraction)
        self._leftAnkleVel = self.ComputeAngVelRel(leftAnkleRotStart, leftAnkleRotEnd, keyFrameDuration, bullet_client)

        ##### Right Hip
        rightHipRotStart = [frameData[20], frameData[21], frameData[22], frameData[23]]
        rightHipRotEnd = [frameDataNext[20], frameDataNext[21], frameDataNext[22], frameDataNext[23]]
        self._rightHipRot = bullet_client.getQuaternionSlerp(rightHipRotStart, rightHipRotEnd, frameFraction)
        self._rightHipVel = self.ComputeAngVelRel(rightHipRotStart, rightHipRotEnd, keyFrameDuration, bullet_client)

        ##### Right Knee
        rightKneeRotStart = [frameData[24], frameData[25], frameData[26], frameData[27]]
        rightKneeRotEnd = [frameDataNext[24], frameDataNext[25], frameDataNext[26], frameDataNext[27]]
        self._rightKneeRot = bullet_client.getQuaternionSlerp(rightKneeRotStart, rightKneeRotEnd, frameFraction)
        self._rightKneeVel = self.ComputeAngVelRel(rightKneeRotStart, rightKneeRotEnd, keyFrameDuration, bullet_client)

        ##### Right Ankle
        rightAnkleRotStart = [frameData[28], frameData[29], frameData[30], frameData[31]]
        rightAnkleRotEnd = [frameDataNext[28], frameDataNext[29], frameDataNext[30], frameDataNext[31]]
        self._rightAnkleRot = bullet_client.getQuaternionSlerp(rightAnkleRotStart, rightAnkleRotEnd, frameFraction)
        self._rightAnkleVel = self.ComputeAngVelRel(rightAnkleRotStart, rightAnkleRotEnd, keyFrameDuration, bullet_client)

        ##### Lower Back
        lowerBackRotStart = [frameData[32], frameData[33], frameData[34], frameData[35]]
        lowerBackRotEnd = [frameDataNext[32], frameDataNext[33], frameDataNext[34], frameDataNext[35]]
        self._lowerBackRot = bullet_client.getQuaternionSlerp(lowerBackRotStart, lowerBackRotEnd, frameFraction)
        self._lowerBackVel = self.ComputeAngVelRel(lowerBackRotStart, lowerBackRotEnd, keyFrameDuration, bullet_client)

        ##### Upper Back
        upperBackRotStart = [frameData[36], frameData[37], frameData[38], frameData[39]]
        upperBackRotEnd = [frameDataNext[36], frameDataNext[37], frameDataNext[38], frameDataNext[39]]
        self._upperBackRot = bullet_client.getQuaternionSlerp(upperBackRotStart, upperBackRotEnd, frameFraction)
        self._upperBackVel = self.ComputeAngVelRel(upperBackRotStart, upperBackRotEnd, keyFrameDuration, bullet_client)

        ##### Chest
        chestRotStart = [frameData[40], frameData[41], frameData[42], frameData[43]]
        chestRotEnd = [frameDataNext[40], frameDataNext[41], frameDataNext[42], frameDataNext[43]]
        self._chestRot = bullet_client.getQuaternionSlerp(chestRotStart, chestRotEnd, frameFraction)
        self._chestVel = self.ComputeAngVelRel(chestRotStart, chestRotEnd, keyFrameDuration, bullet_client)

        ##### Lower Neck
        lowerNeckRotStart = [frameData[44], frameData[45], frameData[46], frameData[47]]
        lowerNeckRotEnd = [frameDataNext[44], frameDataNext[45], frameDataNext[46], frameDataNext[47]]
        self._lowerNeckRot = bullet_client.getQuaternionSlerp(lowerNeckRotStart, lowerNeckRotEnd, frameFraction)
        self._lowerNeckVel = self.ComputeAngVelRel(lowerNeckRotStart, lowerNeckRotEnd, keyFrameDuration, bullet_client)

        ##### Upper Neck
        upperNeckRotStart = [frameData[48], frameData[49], frameData[50], frameData[51]]
        upperNeckRotEnd = [frameDataNext[48], frameDataNext[49], frameDataNext[50], frameDataNext[51]]
        self._upperNeckRot = bullet_client.getQuaternionSlerp(upperNeckRotStart, upperNeckRotEnd, frameFraction)
        self._upperNeckVel = self.ComputeAngVelRel(upperNeckRotStart, upperNeckRotEnd, keyFrameDuration, bullet_client)

        ##### Left Clavicle
        leftClavicleRotStart = [frameData[52], frameData[53], frameData[54], frameData[55]]
        leftClavicleRotEnd = [frameDataNext[52], frameDataNext[53], frameDataNext[54], frameDataNext[55]]
        self._leftClavicleRot = bullet_client.getQuaternionSlerp(leftClavicleRotStart, leftClavicleRotEnd, frameFraction)
        self._leftClavicleVel = self.ComputeAngVelRel(leftClavicleRotStart, leftClavicleRotEnd, keyFrameDuration, bullet_client)

        ##### Left Shoulder
        leftShoulderRotStart = [frameData[56], frameData[57], frameData[58], frameData[59]]
        leftShoulderRotEnd = [frameDataNext[56], frameDataNext[57], frameDataNext[58], frameDataNext[59]]
        self._leftShoulderRot = bullet_client.getQuaternionSlerp(leftShoulderRotStart, leftShoulderRotEnd, frameFraction)
        self._leftShoulderVel = self.ComputeAngVelRel(leftShoulderRotStart, leftShoulderRotEnd, keyFrameDuration, bullet_client)

        ##### Left Elbow
        leftElbowRotStart = [frameData[60], frameData[61], frameData[62], frameData[63]]
        leftElbowRotEnd = [frameDataNext[60], frameDataNext[61], frameDataNext[62], frameDataNext[63]]
        self._leftElbowRot = bullet_client.getQuaternionSlerp(leftElbowRotStart, leftElbowRotEnd, frameFraction)
        self._leftElbowVel = self.ComputeAngVelRel(leftElbowRotStart, leftElbowRotEnd, keyFrameDuration, bullet_client)

        ##### Right Clavicle
        rightClavicleRotStart = [frameData[64], frameData[65], frameData[66], frameData[67]]
        rightClavicleRotEnd = [frameDataNext[64], frameDataNext[65], frameDataNext[66], frameDataNext[67]]
        self._rightClavicleRot = bullet_client.getQuaternionSlerp(rightClavicleRotStart, rightClavicleRotEnd, frameFraction)
        self._rightClavicleVel = self.ComputeAngVelRel(rightClavicleRotStart, rightClavicleRotEnd, keyFrameDuration, bullet_client)

        ##### Right Shoulder
        rightShoulderRotStart = [frameData[68], frameData[69], frameData[70], frameData[71]]
        rightShoulderRotEnd = [frameDataNext[68], frameDataNext[69], frameDataNext[70], frameDataNext[71]]
        self._rightShoulderRot = bullet_client.getQuaternionSlerp(rightShoulderRotStart, rightShoulderRotEnd, frameFraction)
        self._rightShoulderVel = self.ComputeAngVelRel(rightShoulderRotStart, rightShoulderRotEnd, keyFrameDuration, bullet_client)

        ##### Right Elbow
        rightElbowRotStart = [frameData[72], frameData[73], frameData[74], frameData[75]]
        rightElbowRotEnd = [frameDataNext[72], frameDataNext[73], frameDataNext[74], frameDataNext[75]]
        self._rightElbowRot = bullet_client.getQuaternionSlerp(rightElbowRotStart, rightElbowRotEnd, frameFraction)
        self._rightElbowVel = self.ComputeAngVelRel(rightElbowRotStart, rightElbowRotEnd, keyFrameDuration, bullet_client)

        pose = self.GetPose()
        return pose


    def StateConvert(self, state):
        self._basePos = [state[0], state[1], state[2]]
        self._baseOrn = [state[3], state[4], state[5], state[6]]

        self._leftHipRot=[state[7], state[8], state[9], state[10]]
        self._leftKneeRot=[state[11], state[12], state[13], state[14]]
        self._leftAnkleRot=[state[15], state[16], state[17], state[18]]
        self._rightHipRot=[state[19], state[20], state[21], state[22]]
        self._rightKneeRot=[state[23], state[24], state[25], state[26]]
        self._rightAnkleRot=[state[27], state[28], state[29], state[30]]

        self._lowerBackRot=[state[31], state[32], state[33], state[34]]
        self._upperBackRot=[state[35], state[36], state[37], state[38]]
        self._chestRot=[state[39], state[40], state[41], state[42]]
        self._lowerNeckRot=[state[43], state[44], state[45], state[46]]
        self._upperNeckRot=[state[47], state[48], state[49], state[50]]

        self._leftClavicleRot=[state[51], state[52], state[53], state[54]]
        self._leftShoulderRot=[state[55], state[56], state[57], state[58]]
        self._leftElbowRot=[state[59], state[60], state[61], state[62]]
        self._rightClavicleRot=[state[63], state[64], state[65], state[66]]
        self._rightShoulderRot=[state[67], state[68], state[69], state[70]]
        self._rightElbowRot=[state[71], state[72], state[73], state[74]]

        # Velocity
        self._baseLinVel=[state[75], state[76], state[77]]
        self._baseAngVel=[state[78], state[79], state[80]]

        self._leftHipVel=[state[81], state[82], state[83]]
        self._leftKneeVel=[state[84], state[85], state[86]]
        self._leftAnkleVel=[state[87], state[88], state[89]]
        self._rightHipVel=[state[90], state[91], state[92]]
        self._rightKneeVel=[state[93], state[94], state[95]]
        self._rightAnkleVel=[state[96], state[97], state[98]]

        self._lowerBackVel=[state[99], state[100], state[101]]
        self._upperBackVel=[state[102], state[103], state[104]]
        self._chestVel=[state[105], state[106], state[107]]
        self._lowerNeckVel=[state[108], state[109], state[110]]
        self._upperNeckVel=[state[111], state[112], state[113]]

        self._leftClavicleVel=[state[114], state[115], state[116]]
        self._leftShoulderVel=[state[117], state[118], state[119]]
        self._leftElbowVel=[state[120], state[121], state[122]]
        self._rightClavicleVel=[state[123], state[124], state[125]]
        self._rightShoulderVel=[state[126], state[127], state[128]]
        self._rightElbowVel=[state[129], state[130], state[131]]

        pose = self.GetPose()
        return pose

    def GetPose(self):
        pose = list(self._basePos) + list(self._baseOrn) + list(self._leftHipRot) + list(self._leftKneeRot) + list(self._leftAnkleRot) + list(self._rightHipRot) + list(self._rightKneeRot) + list(self._rightAnkleRot) + list(self._lowerBackRot) + list(self._upperBackRot) + list(self._chestRot) + list(self._lowerNeckRot) + list(self._upperNeckRot) + list(self._leftClavicleRot) + list(self._leftShoulderRot) + list(self._leftElbowRot) + list(self._rightClavicleRot) + list(self._rightShoulderRot) + list(self._rightElbowRot)

        return pose

    def buildHeadingTrans(self, rootOrn):
        #align root transform 'forward' with world-space x axis
        eul = self._p.getEulerFromQuaternion(rootOrn)
        refDir = [1, 0, 0]
        rotVec = self._p.rotateVector(rootOrn, refDir)
        heading = math.atan2(-rotVec[2], rotVec[0])
        heading2 = eul[1]
        #print("heading=",heading)
        headingOrn = self._p.getQuaternionFromAxisAngle([0, 1, 0], -heading)
        return headingOrn

    def buildOriginTrans(self):
        rootPos, rootOrn = self._p.getBasePositionAndOrientation(self.object_id[0])

        #print("rootPos=",rootPos, " rootOrn=",rootOrn)
        invRootPos = [-rootPos[0], 0, -rootPos[2]]
        #invOrigTransPos, invOrigTransOrn = self._p.invertTransform(rootPos,rootOrn)
        headingOrn = self.buildHeadingTrans(rootOrn)
        #print("headingOrn=",headingOrn)
        headingMat = self._p.getMatrixFromQuaternion(headingOrn)
        #print("headingMat=",headingMat)
        #dummy, rootOrnWithoutHeading = self._p.multiplyTransforms([0,0,0],headingOrn, [0,0,0], rootOrn)
        #dummy, invOrigTransOrn = self._p.multiplyTransforms([0,0,0],rootOrnWithoutHeading, invOrigTransPos, invOrigTransOrn)

        invOrigTransPos, invOrigTransOrn = self._p.multiplyTransforms([0, 0, 0],
                                                                                    headingOrn,
                                                                                    invRootPos,
                                                                                    [0, 0, 0, 1])
        #print("invOrigTransPos=",invOrigTransPos)
        #print("invOrigTransOrn=",invOrigTransOrn)
        invOrigTransMat = self._p.getMatrixFromQuaternion(invOrigTransOrn)
        #print("invOrigTransMat =",invOrigTransMat )
        return invOrigTransPos, invOrigTransOrn

    def calc_state(self):

        states = self._p.getJointStatesMultiDof(
            self.object_id[0], self._jointIndicesAll
        )
        pose = []
        vels = []
        for state in states:
            if len(state[0]) == 4:
                pos = self._p.getEulerFromQuaternion(state[0])
            else:
                pos = state[0]
            pose += pos
            vels += state[1]

        basePos, baseOrn = self._p.getBasePositionAndOrientation(self.object_id[0])

        self.body_xyz = basePos
        self.body_rpy = self._p.getEulerFromQuaternion(baseOrn)

        if self.initial_y is None:
            self.initial_y = self.body_xyz[1]

        roll, yaw, pitch = self.body_rpy

        rot = np.array(
            [
                [np.cos(-yaw), 0, np.sin(-yaw)],
                [0, 1, 0],
                [-np.sin(-yaw), 0, np.cos(-yaw)],
            ]
        )

        # rot = np.array(
        #     [
        #         [np.cos(-yaw), -np.sin(-yaw), 0],
        #         [np.sin(-yaw), np.cos(-yaw), 0],
        #         [0, 0, 1],
        #     ]
        # )
        body_speed , _ = self._p.getBaseVelocity(self.object_id[0])
        self.body_vel = np.dot(rot, body_speed)
        vx, vy, vz = self.body_vel

        # wx, wy, wz = self.robot_body.angular_speed() / 10

        # more = np.array(
        #     [self.body_xyz[2] - self.initial_z, vx, vy, vz, roll, pitch],
        #     dtype=np.float32,
        # )
        # vx, vy, vz = self.robot_body.speed()
        # more = np.array(
        #     [self.body_xyz[2] - self.initial_z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz],
        #     dtype=np.float32,
        # )


    # def getPhase(self):
    #     keyFrameDuration = self._mocap_data.KeyFrameDuraction
    #     cycleTime = keyFrameDuration * (self._mocap_data.NumFrames - 1)
    #     phase = self._simTime / cycleTime
    #     phase = math.fmod(phase, 1.0)
    #     if (phase < 0):
    #         phase += 1
    #     return phase

    def getState1(self):

        stateVector = []

        rootTransPos, rootTransOrn = self.buildOriginTrans()
        basePos, baseOrn = self._p.getBasePositionAndOrientation(self.object_id[0])

        rootPosRel, dummy = self._p.multiplyTransforms(rootTransPos, rootTransOrn,
                                                                    basePos, [0, 0, 0, 1])

        self.body_xyz = basePos
        self.body_rpy = self._p.getEulerFromQuaternion(baseOrn)
        #print("!!!rootPosRel =",rootPosRel )
        #print("rootTransPos=",rootTransPos)
        #print("basePos=",basePos)
        localPos, localOrn = self._p.multiplyTransforms(rootTransPos, rootTransOrn,
                                                                    basePos, baseOrn)

        localPos = [
            localPos[0] - rootPosRel[0], localPos[1] - rootPosRel[1], localPos[2] - rootPosRel[2]
        ]
        #print("localPos=",localPos)

        stateVector.append(rootPosRel[1])

        #self.pb2dmJoints=[0,1,2,9,10,11,3,4,5,12,13,14,6,7,8]
        self.pb2dmJoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        linkIndicesSim = []
        for pbJoint in range(self._p.getNumJoints(self.object_id[0])):
            linkIndicesSim.append(self.pb2dmJoints[pbJoint])
        
        linkStatesSim = self._p.getLinkStates(self.object_id[0], linkIndicesSim, computeForwardKinematics=True, computeLinkVelocity=True)
        
        for pbJoint in range(self._p.getNumJoints(self.object_id[0])):
            j = self.pb2dmJoints[pbJoint]
            #print("joint order:",j)
            #ls = self._p.getLinkState(self.object_id[0], j, computeForwardKinematics=True)
            ls = linkStatesSim[pbJoint]
            linkPos = ls[0]
            linkOrn = ls[1]
            linkPosLocal, linkOrnLocal = self._p.multiplyTransforms(
                rootTransPos, rootTransOrn, linkPos, linkOrn)
            if (linkOrnLocal[3] < 0):
                linkOrnLocal = [-linkOrnLocal[0], -linkOrnLocal[1], -linkOrnLocal[2], -linkOrnLocal[3]]
            linkPosLocal = [
                linkPosLocal[0] - rootPosRel[0], linkPosLocal[1] - rootPosRel[1],
                linkPosLocal[2] - rootPosRel[2]
            ]
            for l in linkPosLocal:
                stateVector.append(l)
            #re-order the quaternion, DeepMimic uses w,x,y,z

            if (linkOrnLocal[3] < 0):
                linkOrnLocal[0] *= -1
                linkOrnLocal[1] *= -1
                linkOrnLocal[2] *= -1
                linkOrnLocal[3] *= -1

            stateVector.append(linkOrnLocal[3])
            stateVector.append(linkOrnLocal[0])
            stateVector.append(linkOrnLocal[1])
            stateVector.append(linkOrnLocal[2])

        
        for pbJoint in range(self._p.getNumJoints(self.object_id[0])):
            j = self.pb2dmJoints[pbJoint]
            #ls = self._p.getLinkState(self.object_id[0], j, computeLinkVelocity=True)
            ls = linkStatesSim[pbJoint]
            
            linkLinVel = ls[6]
            linkAngVel = ls[7]
            linkLinVelLocal, unused = self._p.multiplyTransforms([0, 0, 0], rootTransOrn,
                                                                                linkLinVel, [0, 0, 0, 1])
            #linkLinVelLocal=[linkLinVelLocal[0]-rootPosRel[0],linkLinVelLocal[1]-rootPosRel[1],linkLinVelLocal[2]-rootPosRel[2]]
            linkAngVelLocal, unused = self._p.multiplyTransforms([0, 0, 0], rootTransOrn,
                                                                                linkAngVel, [0, 0, 0, 1])

            for l in linkLinVelLocal:
                stateVector.append(l)
            for l in linkAngVelLocal:
                stateVector.append(l)

            #print("stateVector len=",len(stateVector))
            #for st in range (len(stateVector)):
            #  print("state[",st,"]=",stateVector[st])
        return stateVector

    def getState(self, kin=False):
        if kin:
            character = self._kin_model
        else:
            character = self.object_id[0]
        self.pb2dmJoints = self._linkIndicesAll[1:]

        sim_basePos, sim_baseOrn = self._p.getBasePositionAndOrientation(character) 
        sim_baseLinVel, sim_baseAngVel = self._p.getBaseVelocity(character)
        sim_jointStates = self._p.getJointStatesMultiDof(character, self.pb2dmJoints)
        
        self.body_xyz = sim_basePos

        simulatedState = StateVector(
            # Position
            basePos = sim_basePos,
            baseOrn = sim_baseOrn,

            leftHipRot = sim_jointStates[lhip][0],
            leftKneeRot = sim_jointStates[lknee][0],
            leftAnkleRot = sim_jointStates[lankle][0],
            rightHipRot = sim_jointStates[rhip][0],
            rightKneeRot = sim_jointStates[rknee][0],
            rightAnkleRot = sim_jointStates[rankle][0],

            lowerBackRot = sim_jointStates[lowerback][0],
            upperBackRot = sim_jointStates[upperback][0],
            chestRot = sim_jointStates[chest][0],
            lowerNeckRot = sim_jointStates[lowerneck][0],
            upperNeckRot = sim_jointStates[upperneck][0],

            leftClavicleRot = sim_jointStates[lclavicle][0],
            leftShoulderRot = sim_jointStates[lshoulder][0],
            leftElbowRot = sim_jointStates[lelbow][0],
            rightClavicleRot = sim_jointStates[rclavicle][0],
            rightShoulderRot = sim_jointStates[rshoulder][0],
            rightElbowRot = sim_jointStates[relbow][0],

            # Velocity
            baseLinVel = sim_baseLinVel,
            baseAngVel = sim_baseAngVel,

            leftHipVel = sim_jointStates[lhip][1],
            leftKneeVel = sim_jointStates[lknee][1],
            leftAnkleVel = sim_jointStates[lankle][1],
            rightHipVel = sim_jointStates[rhip][1],
            rightKneeVel = sim_jointStates[rknee][1],
            rightAnkleVel = sim_jointStates[rankle][1],

            lowerBackVel = sim_jointStates[lowerback][1],
            upperBackVel = sim_jointStates[upperback][1],
            chestVel = sim_jointStates[chest][1],
            lowerNeckVel = sim_jointStates[lowerneck][1],
            upperNeckVel = sim_jointStates[upperneck][1],

            leftClavicleVel = sim_jointStates[lclavicle][1],
            leftShoulderVel = sim_jointStates[lshoulder][1],
            leftElbowVel = sim_jointStates[lelbow][1],
            rightClavicleVel = sim_jointStates[rclavicle][1],
            rightShoulderVel = sim_jointStates[rshoulder][1],
            rightElbowVel = sim_jointStates[relbow][1],
        )

        return simulatedState

    def get_state_array(self, state):
        pos = list(state.basePos) + list(state.baseOrn) \
                 + list(state.leftHipRot) + list(state.leftKneeRot) + list(state.leftAnkleRot) \
                 + list(state.rightHipRot) + list(state.rightKneeRot) + list(state.rightAnkleRot) \
                 + list(state.lowerBackRot) + list(state.upperBackRot) + list(state.chestRot) + list(state.lowerNeckRot) + list(state.upperNeckRot) \
                 + list(state.leftClavicleRot) + list(state.leftShoulderRot) + list(state.leftElbowRot) \
                 + list(state.rightClavicleRot) + list(state.rightShoulderRot) + list(state.rightElbowRot) 
        
        vel = list(state.baseLinVel) + list(state.baseAngVel) \
                     + list(state.leftHipVel) + list(state.leftKneeVel) + list(state.leftAnkleVel) \
                     + list(state.rightHipVel) + list(state.rightKneeVel) + list(state.rightAnkleVel) \
                     + list(state.lowerBackVel) + list(state.upperBackVel) + list(state.chestVel) + list(state.lowerNeckVel) + list(state.upperNeckVel) \
                     + list(state.leftClavicleVel) + list(state.leftShoulderVel) + list(state.leftElbowVel) \
                     + list(state.rightClavicleVel) + list(state.rightShoulderVel) + list(state.rightElbowVel) 

        return np.array(pos + vel)

    def reset(self):
        init_vector = [[0,0.98,0]]
        for i in range(18):
            init_vector.append([0,0,0,1])
        for i in range(19):
            init_vector.append([0,0,0])
        state = StateList2StateVector(init_vector)
        self.initializePose(state, self.object_id[0], initBase=True, initializeVelocity=True)
        self.initializePose(state, self._kin_model, initBase=True, initializeVelocity=True)

        state = self.getState()

        return state

    def computeAngVelRel(self, ornStart, ornEnd, deltaTime, bullet_client):
        ornStartConjugate = [-ornStart[0], -ornStart[1], -ornStart[2], ornStart[3]]
        q_diff = self.quatMul(
            ornStartConjugate,
            ornEnd)  #bullet_client.multiplyTransforms([0,0,0], ornStartConjugate, [0,0,0], ornEnd)

        axis, angle = bullet_client.getAxisAngleFromQuaternion(q_diff)
        angVel = [(axis[0] * angle) / deltaTime, (axis[1] * angle) / deltaTime,
                (axis[2] * angle) / deltaTime]
        return angVel

    def computePDForces(self, desiredPositions, desiredVelocities, maxForces):
        """Compute torques from the PD controller."""
        if desiredVelocities == None:
            desiredVelocities = [0] * 75

        numJoints = len(self._jointIndicesAll)  #self._p.getNumJoints(bodyUniqueId)
        
        curPos, curOrn = self._p.getBasePositionAndOrientation(self.object_id[0])
        q1 = [curPos[0], curPos[1], curPos[2], curOrn[0], curOrn[1], curOrn[2], curOrn[3]]
        #print("q1=",q1)

        #qdot1 = [0,0,0, 0,0,0,0]
        baseLinVel, baseAngVel = self._p.getBaseVelocity(self.object_id[0])
        #print("baseLinVel=",baseLinVel)
        qdot1 = [
            baseLinVel[0], baseLinVel[1], baseLinVel[2], baseAngVel[0], baseAngVel[1], baseAngVel[2], 0
        ]
        #qError = [0,0,0, 0,0,0,0]
        desiredOrn = [
            desiredPositions[3], desiredPositions[4], desiredPositions[5], desiredPositions[6]
        ]
        axis1 = self._p.getAxisDifferenceQuaternion(desiredOrn, curOrn)
        angDiff = self.ComputeAngVel(curOrn, desiredOrn, 1, self._p)
        qError = [
            desiredPositions[0] - curPos[0], desiredPositions[1] - curPos[1],
            desiredPositions[2] - curPos[2], angDiff[0], angDiff[1], angDiff[2], 0
        ]
        target_pos = np.array(desiredPositions)
        #np.savetxt("pb_target_pos.csv", target_pos, delimiter=",")

        qIndex = 7
        qdotIndex = 7
        zeroAccelerations = [0, 0, 0, 0, 0, 0, 0]
        useArray = True
        if useArray:
            jointStates = self._p.getJointStatesMultiDof(self.object_id[0], self._jointIndicesAll)
        

        for i in range(numJoints):
            if useArray:
                js = jointStates[i]
            else:
                js = self._p.getJointStateMultiDof(self.object_id[0], self._jointIndicesAll[i])

            jointPos = js[0]
            jointVel = js[1]
            q1 += jointPos

            if len(js[0]) == 1:
                desiredPos = desiredPositions[qIndex]

                qdiff = desiredPos - jointPos[0]
                qError.append(qdiff)
                zeroAccelerations.append(0.)
                qdot1 += jointVel
                qIndex += 1
                qdotIndex += 1
            if len(js[0]) == 4:
                desiredPos = [
                    desiredPositions[qIndex], desiredPositions[qIndex + 1], desiredPositions[qIndex + 2],
                    desiredPositions[qIndex + 3]
                ]
                #axis = self._p.getAxisDifferenceQuaternion(desiredPos,jointPos)
                angDiff = self.computeAngVelRel(jointPos, desiredPos, 1, self._p)
                #angDiff = self._p.computeAngVelRel(jointPos, desiredPos, 1)

                jointVelNew = [jointVel[0], jointVel[1], jointVel[2], 0]
                qdot1 += jointVelNew
                qError.append(angDiff[0])
                qError.append(angDiff[1])
                qError.append(angDiff[2])
                qError.append(0)
                desiredVel = [
                    desiredVelocities[qdotIndex], desiredVelocities[qdotIndex + 1],
                    desiredVelocities[qdotIndex + 2]
                ]
                zeroAccelerations += [0., 0., 0., 0.]
                qIndex += 4
                qdotIndex += 4


        q = np.array(q1)

        qerr = np.array(qError)

        #np.savetxt("pb_qerro.csv",qerr,delimiter=",")

        #np.savetxt("pb_q.csv", q, delimiter=",")

        qdot = np.array(qdot1)
        #np.savetxt("qdot.csv", qdot, delimiter=",")

        qdotdesired = np.array(desiredVelocities)
        qdoterr = qdotdesired - qdot

        Kp = np.diagflat(self.P)
        Kd = np.diagflat(self.D)

        p = Kp.dot(qError)

        #np.savetxt("pb_qError.csv", qError, delimiter=",")
        #np.savetxt("pb_p.csv", p, delimiter=",")

        d = Kd.dot(qdoterr)

        #np.savetxt("pb_d.csv", d, delimiter=",")
        #np.savetxt("pbqdoterr.csv", qdoterr, delimiter=",")

        M1 = self._p.calculateMassMatrix(self.object_id[0], q1, flags=1)

        M2 = np.array(M1)
        #np.savetxt("M2.csv", M2, delimiter=",")

        M = (M2 + Kd * self.timestep)

        #np.savetxt("pbM_tKd.csv",M, delimiter=",")

        c1 = self._p.calculateInverseDynamics(self.object_id[0], q1, qdot1, zeroAccelerations, flags=1)

        c = np.array(c1)
        #np.savetxt("pb_C.csv",c, delimiter=",")
        A = M
        #p = [0]*43
        #np.savetxt("pb_kp_dot_qError.csv", p)
        #np.savetxt("pb_kd_dot_vError.csv", d)

        b = p + d - c
        #np.savetxt("pb_b_acc.csv",b, delimiter=",")

        useNumpySolver = True
        if useNumpySolver:
            qddot = np.linalg.solve(A, b)
        else:
            qddot = self._p.ldltSolve(self.object_id[0], jointPositions=q1, b=b.tolist(), kd=self.D, t=self.timestep)

        tau = p + d - Kd.dot(qddot) * self.timestep
        #print("len(tau)=",len(tau))
        #np.savetxt("pb_tau_not_clamped.csv", tau, delimiter=",")

        maxF = np.array(maxForces)
        #print("maxF",maxF)
        forces = np.clip(tau, -maxF, maxF)

        #np.savetxt("pb_tau_clamped.csv", tau, delimiter=",")
        return forces


    def applyPDForces(self, taus):
        """Apply pre-computed torques."""
        dofIndex = 7
        scaling = 1
        useArray = True
        indices = []
        forces = []
            
        if (useArray):
            for index in range(len(self._jointIndicesAll)):
                jointIndex = self._jointIndicesAll[index]
                indices.append(jointIndex)
                if self._jointDofCounts[index] == 4:
                    force = [
                        scaling * taus[dofIndex + 0], scaling * taus[dofIndex + 1],
                        scaling * taus[dofIndex + 2]
                    ]
                if self._jointDofCounts[index] == 1:
                    force = [scaling * taus[dofIndex]]
                    #print("force[", jointIndex,"]=",force)
                forces.append(force)
                dofIndex += self._jointDofCounts[index]
            self._p.setJointMotorControlMultiDofArray(self.object_id[0],
                                                                    indices,
                                                                    self._p.TORQUE_CONTROL,
                                                                    forces=forces)
            self._p.applyExternalForce(self.object_id[0], -1, taus[:3], [0.,0.,0.], flags=self._p.LINK_FRAME)
            self._p.applyExternalTorque(self.object_id[0], -1, taus[3:6], flags=self._p.LINK_FRAME)
        else:
            for index in range(len(self._jointIndicesAll)):
                jointIndex = self._jointIndicesAll[index]
                if self._jointDofCounts[index] == 4:
                    force = [
                        scaling * taus[dofIndex + 0], scaling * taus[dofIndex + 1],
                        scaling * taus[dofIndex + 2]
                    ]
                    #print("force[", jointIndex,"]=",force)
                    self._p.setJointMotorControlMultiDof(self.object_id[0],
                                                                        jointIndex,
                                                                        self._p.TORQUE_CONTROL,
                                                                        force=force)
                if self._jointDofCounts[index] == 1:
                    force = [scaling * taus[dofIndex]]
                    #print("force[", jointIndex,"]=",force)
                    self._p.setJointMotorControlMultiDof(
                        self.object_id[0],
                        jointIndex,
                        controlMode=self._p.TORQUE_CONTROL,
                        force=force)
                dofIndex += self._jointDofCounts[index]

    def apply_action(self, tar_pose, ext_force):
        
        assert len(tar_pose) == self.action_dim + 7

        tar_pose = np.array(tar_pose)
        self._p.resetBasePositionAndOrientation(
            self.object_id[0],
            tar_pose[:3].tolist(),
            tar_pose[3:7].tolist(),
        )

        usePythonStablePD = False
        if usePythonStablePD:
            taus = self.computePDForces(tar_pose,
                                                desiredVelocities=None,
                                                maxForces=self.maxForces)

            if ext_force is not None:
                self._p.applyExternalForce(self.object_id[0], -1, ext_force, [0.,0.,0.], flags=self._p.LINK_FRAME)
                
            self.applyPDForces(taus)
        else:
            # taus = self.computePDForces(tar_pose, desiredVelocities=None, maxForces=self.maxForces)

            self.computeAndApplyPDForces(tar_pose, self.maxForces)

            # self._p.applyExternalForce(self.object_id[0], -1, taus[:3], [0.,0.,0.], flags=self._p.LINK_FRAME)
            # self._p.applyExternalTorque(self.object_id[0], -1, taus[3:6], flags=self._p.LINK_FRAME)

    def computeAndApplyPDForces(self, desiredPositions, maxForces):
        dofIndex = 7
        scaling = 1
        indices = [] 
        forces = []
        targetPositions=[]
        targetVelocities=[]
        kps = []
        kds = []
        
        for index in range(len(self._jointIndicesAll)):
            jointIndex = self._jointIndicesAll[index]
            indices.append(jointIndex)
            kps.append(self.P[dofIndex])
            kds.append(self.D[dofIndex])
            if self._jointDofCounts[index] == 4:
                force = [
                    scaling * maxForces[dofIndex + 0],
                    scaling * maxForces[dofIndex + 1],
                    scaling * maxForces[dofIndex + 2]
                ]
                targetVelocity = [0,0,0]
                targetPosition = [
                    desiredPositions[dofIndex + 0],
                    desiredPositions[dofIndex + 1],
                    desiredPositions[dofIndex + 2],
                    desiredPositions[dofIndex + 3]
                ]
            if self._jointDofCounts[index] == 1:
                force = [scaling * maxForces[dofIndex]]
                targetPosition = [desiredPositions[dofIndex+0]]
                targetVelocity = [0]
            forces.append(force)
            targetPositions.append(targetPosition)
            targetVelocities.append(targetVelocity)
            dofIndex += self._jointDofCounts[index]
            
        #static char* kwlist[] = { "bodyUniqueId", 
        #"jointIndices", 
        #"controlMode", "targetPositions", "targetVelocities", "forces", "positionGains", "velocityGains", "maxVelocities", "physicsClientId", NULL };
        self._p.setJointMotorControlMultiDofArray(self.object_id[0],
                                                            indices,
                                                            self._p.STABLE_PD_CONTROL,
                                                            targetPositions = targetPositions,
                                                            targetVelocities = targetVelocities,
                                                            forces=forces,
                                                            positionGains = kps,
                                                            velocityGains = kds,
                                                            )



    def alive_bonus(self, z):
        return +2 if z > 0.17 else -1   # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying

    def calc_feet_state(self):
        for i, j in enumerate(self.feet_jointsind):
            self.feet_xyz[i] = self._p.getJointState(self.object_id[0], j)[0]

    def ComputeLinVel(self, posStart, posEnd, deltaTime):
        vel = [(posEnd[0] - posStart[0]) / deltaTime, (posEnd[1] - posStart[1]) / deltaTime,
            (posEnd[2] - posStart[2]) / deltaTime]
        return vel

    def ComputeAngVel(self, ornStart, ornEnd, deltaTime, bullet_client):
        dorn = bullet_client.getDifferenceQuaternion(ornStart, ornEnd)
        axis, angle = bullet_client.getAxisAngleFromQuaternion(dorn)
        angVel = [(axis[0] * angle) / deltaTime, (axis[1] * angle) / deltaTime,
                (axis[2] * angle) / deltaTime]
        return angVel

    def ComputeAngVelRel(self, ornStart, ornEnd, deltaTime, bullet_client):
        ornStartConjugate = [-ornStart[0], -ornStart[1], -ornStart[2], ornStart[3]]
        pos_diff, q_diff = bullet_client.multiplyTransforms([0, 0, 0], ornStartConjugate, [0, 0, 0],
                                                            ornEnd)
        axis, angle = bullet_client.getAxisAngleFromQuaternion(q_diff)
        angVel = [(axis[0] * angle) / deltaTime, (axis[1] * angle) / deltaTime,
                (axis[2] * angle) / deltaTime]
        return angVel

    def quatMul(self, q1, q2):
        return [
            q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1],
            q1[3] * q2[1] + q1[1] * q2[3] + q1[2] * q2[0] - q1[0] * q2[2],
            q1[3] * q2[2] + q1[2] * q2[3] + q1[0] * q2[1] - q1[1] * q2[0],
            q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2]
        ]

    def calcRootRotDiff(self, orn0, orn1):
        orn0Conj = [-orn0[0], -orn0[1], -orn0[2], orn0[3]]
        q_diff = self.quatMul(orn1, orn0Conj)
        axis, angle = self._p.getAxisAngleFromQuaternion(q_diff)
        return angle * angle

    def calcRootAngVelErr(self, vel0, vel1):
        diff = [vel0[0] - vel1[0], vel0[1] - vel1[1], vel0[2] - vel1[2]]
        return diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]

    def initializePose(self, state, phys_model, initBase, initializeVelocity=True):
        
        useArray = True
        if initializeVelocity:
            if initBase:
                self._p.resetBasePositionAndOrientation(phys_model, state.basePos,
                                                                    state.baseOrn)
                self._p.resetBaseVelocity(phys_model, state.baseLinVel, state.baseAngVel)
            if useArray:
                indices = self._jointIndicesAll
                jointPositions = [state.leftHipRot, state.leftKneeRot, state.leftAnkleRot, state.rightHipRot, state.rightKneeRot, state.rightAnkleRot, state.lowerBackRot, state.upperBackRot, state.chestRot, state.lowerNeckRot, state.upperNeckRot, state.leftClavicleRot, state.leftShoulderRot, state.leftElbowRot, state.rightClavicleRot, state.rightShoulderRot, state.rightElbowRot]

                jointVelocities = [state.leftHipVel, state.leftKneeVel, state.leftAnkleVel, state.rightHipVel, state.rightKneeVel, state.rightAnkleVel, state.lowerBackVel, state.upperBackVel, state.chestVel, state.lowerNeckVel, state.upperNeckVel, state.leftClavicleVel, state.leftShoulderVel, state.leftElbowVel, state.rightClavicleVel, state.rightShoulderVel, state.rightElbowVel]
                self._p.resetJointStatesMultiDof(phys_model, indices,
                                                            jointPositions, jointVelocities)
            else:
                self._p.resetJointStateMultiDof(phys_model, chest, state.chestRot,
                                                            state.chestVel)
                self._p.resetJointStateMultiDof(phys_model, neck, state.neckRot, state.neckVel)
                self._p.resetJointStateMultiDof(phys_model, rightHip, state.rightHipRot,
                                                            state.rightHipVel)
                self._p.resetJointStateMultiDof(phys_model, rightKnee, state.rightKneeRot,
                                                            state.rightKneeVel)
                self._p.resetJointStateMultiDof(phys_model, rightAnkle, state.rightAnkleRot,
                                                            state.rightAnkleVel)
                self._p.resetJointStateMultiDof(phys_model, rightShoulder,
                                                            state.rightShoulderRot, state.rightShoulderVel)
                self._p.resetJointStateMultiDof(phys_model, rightElbow, state.rightElbowRot,
                                                            state.rightElbowVel)
                self._p.resetJointStateMultiDof(phys_model, leftHip, state.leftHipRot,
                                                            state.leftHipVel)
                self._p.resetJointStateMultiDof(phys_model, leftKnee, state.leftKneeRot,
                                                            state.leftKneeVel)
                self._p.resetJointStateMultiDof(phys_model, leftAnkle, state.leftAnkleRot,
                                                            state.leftAnkleVel)
                self._p.resetJointStateMultiDof(phys_model, leftShoulder,
                                                            state.leftShoulderRot, state.leftShoulderVel)
                self._p.resetJointStateMultiDof(phys_model, leftElbow, state.leftElbowRot,
                                                            state.leftElbowVel)
        else:
        
            if initBase:
                self._p.resetBasePositionAndOrientation(phys_model, state.basePos,
                                                                    state.baseOrn)
            if useArray:
                indices = self._jointIndicesAll
                jointPositions = [state.leftHipRot, state.leftKneeRot, state.leftAnkleRot, state.rightHipRot, state.rightKneeRot, state.rightAnkleRot, state.lowerBackRot, state.upperBackRot, state.chestRot, state.lowerNeckRot, state.upperNeckRot, state.leftClavicleRot, state.leftShoulderRot, state.leftElbowRot, state.rightClavicleRot, state.rightShoulderRot, state.rightElbowRot]
                self._p.resetJointStatesMultiDof(phys_model, indices, jointPositions)
                
            else:
                self._p.resetJointStateMultiDof(phys_model, chest, state.chestRot, [0, 0, 0])
                self._p.resetJointStateMultiDof(phys_model, neck, state.neckRot, [0, 0, 0])
                self._p.resetJointStateMultiDof(phys_model, rightHip, state.rightHipRot,
                                                            [0, 0, 0])
                self._p.resetJointStateMultiDof(phys_model, rightKnee, state.rightKneeRot, [0])
                self._p.resetJointStateMultiDof(phys_model, rightAnkle, state.rightAnkleRot,
                                                            [0, 0, 0])
                self._p.resetJointStateMultiDof(phys_model, rightShoulder,
                                                            state.rightShoulderRot, [0, 0, 0])
                self._p.resetJointStateMultiDof(phys_model, rightElbow, state.rightElbowRot,
                                                            [0])
                self._p.resetJointStateMultiDof(phys_model, leftHip, state.leftHipRot,
                                                            [0, 0, 0])
                self._p.resetJointStateMultiDof(phys_model, leftKnee, state.leftKneeRot, [0])
                self._p.resetJointStateMultiDof(phys_model, leftAnkle, state.leftAnkleRot,
                                                            [0, 0, 0])
                self._p.resetJointStateMultiDof(phys_model, leftShoulder,
                                                            state.leftShoulderRot, [0, 0, 0])
                self._p.resetJointStateMultiDof(phys_model, leftElbow, state.leftElbowRot, [0])
