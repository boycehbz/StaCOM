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
from vclrl_envs.bullet_utils import BodyPart, Joint
import math

chest = 1
neck = 2
rightHip = 3
rightKnee = 4
rightAnkle = 5
rightShoulder = 6
rightElbow = 7
rightWrist = 8
leftHip = 9
leftKnee = 10
leftAnkle = 11
leftShoulder = 12
leftElbow = 13
leftWrist = 14
jointFrictionForce = 0


from collections import namedtuple
State = namedtuple('State', [
    # Position
    'basePos',
    'baseOrn',
    'chestRot', 
    'neckRot', 
    'rightHipRot', 
    'rightKneeRot', 
    'rightAnkleRot', 
    'rightShoulderRot', 
    'rightElbowRot', 
    'leftHipRot', 
    'leftKneeRot', 
    'leftAnkleRot', 
    'leftShoulderRot', 
    'leftElbowRot',

    # Velocity
    'baseLinVel',
    'baseAngVel',
    'chestVel', 
    'neckVel', 
    'rightHipVel', 
    'rightKneeVel', 
    'rightAnkleVel', 
    'rightShoulderVel', 
    'rightElbowVel', 
    'leftHipVel', 
    'leftKneeVel', 
    'leftAnkleVel', 
    'leftShoulderVel', 
    'leftElbowVel',
])


class HumanoidWalker():

    foot_names = ["right_ankle", "left_ankle"]

    def __init__(self, bc):
        self._p = bc
        self.power = 1.0
        
        self.action_dim = 36
        high = np.inf * np.ones(self.action_dim)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # globals + angles + speeds + contacts
        self.state_dim = 77 #6 + self.action_dim * 2 + 2
        high = np.inf * np.ones(self.state_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self._jointDofCounts = [4, 4, 4, 1, 4, 4, 1, 4, 1, 4, 4, 1]

        self.timestep = 1 / 240.
        # maxForce = 1000.0
        # self.maxForces = np.array([maxForce]*43)
        
        # self.P = np.array([1000]*43)
        # self.D = self.P / 10.0


        self.P = np.array([
            0, 0, 0, 0, 0, 0, 0, 1000, 1000, 1000, 1000, 100, 100, 100, 100, 500, 500, 500, 500, 500,
            400, 400, 400, 400, 400, 400, 400, 400, 300, 500, 500, 500, 500, 500, 400, 400, 400, 400,
            400, 400, 400, 400, 300
        ])
        self.D = self.P / 10.0

        self.maxForces = [
            0, 0, 0, 0, 0, 0, 0, 200, 200, 200, 200, 50, 50, 50, 50, 200, 200, 200, 200, 150, 90,
            90, 90, 90, 100, 100, 100, 100, 60, 200, 200, 200, 200, 150, 90, 90, 90, 90, 100, 100,
            100, 100, 60
        ]
        self._linkIndicesAll = [0, chest, neck, rightHip, rightKnee, rightAnkle, rightShoulder, rightElbow, rightWrist, leftHip, leftKnee,
            leftAnkle, leftShoulder, leftElbow, leftWrist
        ]
        self._jointIndicesAll = [
            chest, neck, rightHip, rightKnee, rightAnkle, rightShoulder, rightElbow, leftHip, leftKnee,
            leftAnkle, leftShoulder, leftElbow
        ]
        self._end_effectors = [rightAnkle, rightWrist, leftAnkle, leftWrist]  #ankle and wrist, both left and right
        self.feet_jointsind = [rightAnkle, leftAnkle]
        self.feet_xyz = np.zeros((len(self.foot_names), 3))

        self.initial_y = None

    def set_mocap(self, _mocapdata):
        self._mocap_data = _mocapdata

    def load_robot_model(self):
        flags = (
            self._p.URDF_MAINTAIN_LINK_ORDER |
            self._p.URDF_USE_SELF_COLLISION
            | self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        )
        model_path = os.path.join("vclrl_envs", "data", "humanoid.urdf")

        self.base_position = (0, 0.889540259, 0)
        self.base_orientation = (0, 0, 0, 1)

        self.object_id = (self._p.loadURDF(model_path, 
                                            self.base_position,
                                            self.base_orientation,
                                            globalScaling=0.25,
                                            useFixedBase=False,
                                            flags=flags),)

        self._kin_model = self._p.loadURDF(
            model_path, [0, 0.85, 0],
            globalScaling=0.25,
            useFixedBase=True,
            flags=self._p.URDF_MAINTAIN_LINK_ORDER)

        # Change Dynamics
        self._p.changeDynamics(self.object_id[0], -1, lateralFriction=0.9)
        for j in range(self._p.getNumJoints(self.object_id[0])):
            self._p.changeDynamics(self.object_id[0], j, lateralFriction=0.9)

        self._p.changeDynamics(self.object_id[0], -1, linearDamping=0, angularDamping=0)
        self._p.changeDynamics(self._kin_model, -1, linearDamping=0, angularDamping=0)

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
        self.init(basePos=(0, 0.889540259, 0))
        # T-pose
        init_action = np.zeros(self.action_dim)
        self.base_joint_angles = self.ConvertFromAction(init_action)

    def init(self,
                basePos=[0, 0, 0],
                baseOrn=[0, 0, 0, 1],
                chestRot=[0, 0, 0, 1],
                neckRot=[0, 0, 0, 1],
                rightHipRot=[0, 0, 0, 1],
                rightKneeRot=[0],
                rightAnkleRot=[0, 0, 0, 1],
                rightShoulderRot=[0, 0, 0, 1],
                rightElbowRot=[0],
                leftHipRot=[0, 0, 0, 1],
                leftKneeRot=[0],
                leftAnkleRot=[0, 0, 0, 1],
                leftShoulderRot=[0, 0, 0, 1],
                leftElbowRot=[0],
                baseLinVel=[0, 0, 0],
                baseAngVel=[0, 0, 0],
                chestVel=[0, 0, 0],
                neckVel=[0, 0, 0],
                rightHipVel=[0, 0, 0],
                rightKneeVel=[0],
                rightAnkleVel=[0, 0, 0],
                rightShoulderVel=[0, 0, 0],
                rightElbowVel=[0],
                leftHipVel=[0, 0, 0],
                leftKneeVel=[0],
                leftAnkleVel=[0, 0, 0],
                leftShoulderVel=[0, 0, 0],
                leftElbowVel=[0]):

        self._basePos = basePos
        self._baseLinVel = baseLinVel
        #print("HumanoidPoseInterpolator.Reset: baseLinVel = ", baseLinVel)
        self._baseOrn = baseOrn
        self._baseAngVel = baseAngVel

        self._chestRot = chestRot
        self._chestVel = chestVel
        self._neckRot = neckRot
        self._neckVel = neckVel

        self._rightHipRot = rightHipRot
        self._rightHipVel = rightHipVel
        self._rightKneeRot = rightKneeRot
        self._rightKneeVel = rightKneeVel
        self._rightAnkleRot = rightAnkleRot
        self._rightAnkleVel = rightAnkleVel

        self._rightShoulderRot = rightShoulderRot
        self._rightShoulderVel = rightShoulderVel
        self._rightElbowRot = rightElbowRot
        self._rightElbowVel = rightElbowVel

        self._leftHipRot = leftHipRot
        self._leftHipVel = leftHipVel
        self._leftKneeRot = leftKneeRot
        self._leftKneeVel = leftKneeVel
        self._leftAnkleRot = leftAnkleRot
        self._leftAnkleVel = leftAnkleVel

        self._leftShoulderRot = leftShoulderRot
        self._leftShoulderVel = leftShoulderVel
        self._leftElbowRot = leftElbowRot
        self._leftElbowVel = leftElbowVel


    # def ConvertFromAction1(self, action):
    #     #turn action into pose
    #     index = 0
    #     angle = [action[index], action[index + 1], action[index + 2]]
    #     index += 3
    #     self._chestRot = self._p.getQuaternionFromEuler(angle)
    #     #print("pose._chestRot=",pose._chestRot)

    #     angle = [action[index], action[index + 1], action[index + 2]]
    #     index += 3
    #     self._neckRot = self._p.getQuaternionFromEuler(angle)

    #     angle = [action[index + 0], action[index + 1], action[index + 2]]
    #     index += 3
    #     self._rightHipRot = self._p.getQuaternionFromEuler(angle)

    #     angle = action[index]
    #     index += 1
    #     self._rightKneeRot = [angle]

    #     angle = [action[index + 0], action[index + 1], action[index + 2]]
    #     index += 3
    #     self._rightAnkleRot = self._p.getQuaternionFromEuler(angle)

    #     angle = [action[index + 0], action[index + 1], action[index + 2]]
    #     index += 3
    #     self._rightShoulderRot = self._p.getQuaternionFromEuler(angle)

    #     angle = action[index]
    #     index += 1
    #     self._rightElbowRot = [angle]

    #     angle = [action[index + 0], action[index + 1], action[index + 2]]
    #     index += 3
    #     self._leftHipRot = self._p.getQuaternionFromEuler(angle)

    #     angle = action[index]
    #     index += 1
    #     self._leftKneeRot = [angle]

    #     angle = [action[index + 0], action[index + 1], action[index + 2]]
    #     index += 3
    #     self._leftAnkleRot = self._p.getQuaternionFromEuler(angle)

    #     angle = [action[index + 0], action[index + 1], action[index + 2]]
    #     index += 3
    #     self._leftShoulderRot = self._p.getQuaternionFromEuler(angle)

    #     angle = action[index]
    #     index += 1
    #     self._leftElbowRot = [angle]

    #     pose = self.GetPose()
    #     return pose

    def ConvertFromAction(self, action):
        # 36 dim action
        #turn action into pose

        index = 0
        angle = action[index + 3]
        axis = [action[index + 0], action[index + 1], action[index + 2]]
        index += 4
        self._chestRot = self._p.getQuaternionFromAxisAngle(axis, angle)
        #print("pose._chestRot=",pose._chestRot)

        angle = action[index + 3]
        axis = [action[index + 0], action[index + 1], action[index + 2]]
        index += 4
        self._neckRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        angle = action[index + 3]
        axis = [action[index + 0], action[index + 1], action[index + 2]]
        index += 4
        self._rightHipRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        angle = action[index]
        index += 1
        self._rightKneeRot = [angle]

        angle = action[index + 3]
        axis = [action[index + 0], action[index + 1], action[index + 2]]
        index += 4
        self._rightAnkleRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        angle = action[index + 3]
        axis = [action[index + 0], action[index + 1], action[index + 2]]
        index += 4
        self._rightShoulderRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        angle = action[index]
        index += 1
        self._rightElbowRot = [angle]

        angle = action[index + 3]
        axis = [action[index + 0], action[index + 1], action[index + 2]]
        index += 4
        self._leftHipRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        angle = action[index]
        index += 1
        self._leftKneeRot = [angle]

        angle = action[index + 3]
        axis = [action[index + 0], action[index + 1], action[index + 2]]
        index += 4
        self._leftAnkleRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        angle = action[index + 3]
        axis = [action[index + 0], action[index + 1], action[index + 2]]
        index += 4
        self._leftShoulderRot = self._p.getQuaternionFromAxisAngle(axis, angle)

        angle = action[index]
        index += 1
        self._leftElbowRot = [angle]

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
        frameData = _mocap_data._motion_data['Frames'][_frame]
        frameDataNext = _mocap_data._motion_data['Frames'][_frameNext]
        
        self.Slerp(frameFraction, frameData, frameDataNext, self._p)
        #print("self._poseInterpolator.Slerp(", frameFraction,")=", pose)
        self.computeCycleOffset(_mocap_data)
        oldPos = self._basePos
        self._basePos = [
            oldPos[0] + _cycleCount * self._cycleOffset[0],
            oldPos[1] + _cycleCount * self._cycleOffset[1],
            oldPos[2] + _cycleCount * self._cycleOffset[2]
        ]
        pose = self.GetPose()

        return pose

        # self.Slerp(frameFraction, frameData, frameDataNext, self._p)

        # pose = self.GetPose()

        # return pose

    def Slerp(self, frameFraction, frameData, frameDataNext, bullet_client):
        keyFrameDuration = frameData[0]
        basePos1Start = [frameData[1], frameData[2], frameData[3]]
        basePos1End = [frameDataNext[1], frameDataNext[2], frameDataNext[3]]
        self._basePos = [
            basePos1Start[0] + frameFraction * (basePos1End[0] - basePos1Start[0]),
            basePos1Start[1] + frameFraction * (basePos1End[1] - basePos1Start[1]),
            basePos1Start[2] + frameFraction * (basePos1End[2] - basePos1Start[2])
        ]
        self._baseLinVel = self.ComputeLinVel(basePos1Start, basePos1End, keyFrameDuration)
        baseOrn1Start = [frameData[5], frameData[6], frameData[7], frameData[4]]
        baseOrn1Next = [frameDataNext[5], frameDataNext[6], frameDataNext[7], frameDataNext[4]]
        self._baseOrn = bullet_client.getQuaternionSlerp(baseOrn1Start, baseOrn1Next, frameFraction)
        self._baseAngVel = self.ComputeAngVel(baseOrn1Start, baseOrn1Next, keyFrameDuration,
                                            bullet_client)

        ##pre-rotate to make z-up
        #y2zPos=[0,0,0.0]
        #y2zOrn = p.getQuaternionFromEuler([1.57,0,0])
        #basePos,baseOrn = p.multiplyTransforms(y2zPos, y2zOrn,basePos1,baseOrn1)

        chestRotStart = [frameData[9], frameData[10], frameData[11], frameData[8]]
        chestRotEnd = [frameDataNext[9], frameDataNext[10], frameDataNext[11], frameDataNext[8]]
        self._chestRot = bullet_client.getQuaternionSlerp(chestRotStart, chestRotEnd, frameFraction)
        self._chestVel = self.ComputeAngVelRel(chestRotStart, chestRotEnd, keyFrameDuration,
                                            bullet_client)

        neckRotStart = [frameData[13], frameData[14], frameData[15], frameData[12]]
        neckRotEnd = [frameDataNext[13], frameDataNext[14], frameDataNext[15], frameDataNext[12]]
        self._neckRot = bullet_client.getQuaternionSlerp(neckRotStart, neckRotEnd, frameFraction)
        self._neckVel = self.ComputeAngVelRel(neckRotStart, neckRotEnd, keyFrameDuration,
                                            bullet_client)

        rightHipRotStart = [frameData[17], frameData[18], frameData[19], frameData[16]]
        rightHipRotEnd = [frameDataNext[17], frameDataNext[18], frameDataNext[19], frameDataNext[16]]
        self._rightHipRot = bullet_client.getQuaternionSlerp(rightHipRotStart, rightHipRotEnd,
                                                            frameFraction)
        self._rightHipVel = self.ComputeAngVelRel(rightHipRotStart, rightHipRotEnd, keyFrameDuration,
                                                bullet_client)

        rightKneeRotStart = [frameData[20]]
        rightKneeRotEnd = [frameDataNext[20]]
        self._rightKneeRot = [
            rightKneeRotStart[0] + frameFraction * (rightKneeRotEnd[0] - rightKneeRotStart[0])
        ]
        self._rightKneeVel = [(rightKneeRotEnd[0] - rightKneeRotStart[0]) / keyFrameDuration]

        rightAnkleRotStart = [frameData[22], frameData[23], frameData[24], frameData[21]]
        rightAnkleRotEnd = [frameDataNext[22], frameDataNext[23], frameDataNext[24], frameDataNext[21]]
        self._rightAnkleRot = bullet_client.getQuaternionSlerp(rightAnkleRotStart, rightAnkleRotEnd,
                                                            frameFraction)
        self._rightAnkleVel = self.ComputeAngVelRel(rightAnkleRotStart, rightAnkleRotEnd,
                                                    keyFrameDuration, bullet_client)

        rightShoulderRotStart = [frameData[26], frameData[27], frameData[28], frameData[25]]
        rightShoulderRotEnd = [
            frameDataNext[26], frameDataNext[27], frameDataNext[28], frameDataNext[25]
        ]
        self._rightShoulderRot = bullet_client.getQuaternionSlerp(rightShoulderRotStart,
                                                                rightShoulderRotEnd, frameFraction)
        self._rightShoulderVel = self.ComputeAngVelRel(rightShoulderRotStart, rightShoulderRotEnd,
                                                    keyFrameDuration, bullet_client)

        rightElbowRotStart = [frameData[29]]
        rightElbowRotEnd = [frameDataNext[29]]
        self._rightElbowRot = [
            rightElbowRotStart[0] + frameFraction * (rightElbowRotEnd[0] - rightElbowRotStart[0])
        ]
        self._rightElbowVel = [(rightElbowRotEnd[0] - rightElbowRotStart[0]) / keyFrameDuration]

        leftHipRotStart = [frameData[31], frameData[32], frameData[33], frameData[30]]
        leftHipRotEnd = [frameDataNext[31], frameDataNext[32], frameDataNext[33], frameDataNext[30]]
        self._leftHipRot = bullet_client.getQuaternionSlerp(leftHipRotStart, leftHipRotEnd,
                                                            frameFraction)
        self._leftHipVel = self.ComputeAngVelRel(leftHipRotStart, leftHipRotEnd, keyFrameDuration,
                                                bullet_client)

        leftKneeRotStart = [frameData[34]]
        leftKneeRotEnd = [frameDataNext[34]]
        self._leftKneeRot = [
            leftKneeRotStart[0] + frameFraction * (leftKneeRotEnd[0] - leftKneeRotStart[0])
        ]
        self._leftKneeVel = [(leftKneeRotEnd[0] - leftKneeRotStart[0]) / keyFrameDuration]

        leftAnkleRotStart = [frameData[36], frameData[37], frameData[38], frameData[35]]
        leftAnkleRotEnd = [frameDataNext[36], frameDataNext[37], frameDataNext[38], frameDataNext[35]]
        self._leftAnkleRot = bullet_client.getQuaternionSlerp(leftAnkleRotStart, leftAnkleRotEnd,
                                                            frameFraction)
        self._leftAnkleVel = self.ComputeAngVelRel(leftAnkleRotStart, leftAnkleRotEnd,
                                                keyFrameDuration, bullet_client)

        leftShoulderRotStart = [frameData[40], frameData[41], frameData[42], frameData[39]]
        leftShoulderRotEnd = [
            frameDataNext[40], frameDataNext[41], frameDataNext[42], frameDataNext[39]
        ]
        self._leftShoulderRot = bullet_client.getQuaternionSlerp(leftShoulderRotStart,
                                                                leftShoulderRotEnd, frameFraction)
        self._leftShoulderVel = self.ComputeAngVelRel(leftShoulderRotStart, leftShoulderRotEnd,
                                                    keyFrameDuration, bullet_client)

        leftElbowRotStart = [frameData[43]]
        leftElbowRotEnd = [frameDataNext[43]]
        self._leftElbowRot = [
            leftElbowRotStart[0] + frameFraction * (leftElbowRotEnd[0] - leftElbowRotStart[0])
        ]
        self._leftElbowVel = [(leftElbowRotEnd[0] - leftElbowRotStart[0]) / keyFrameDuration]

        pose = self.GetPose()
        return pose


    def StateConvert(self, state):
        self._basePos = [state[0], state[1], state[2]]
        self._baseOrn = [state[3], state[4], state[5], state[6]]
        self._chestRot = [state[7], state[8], state[9], state[10]]
        self._neckRot = [state[11], state[12], state[13], state[14]]
        self._rightHipRot = [state[15], state[16], state[17], state[18]]
        self._rightKneeRot = [state[19]]
        self._rightAnkleRot = [state[20], state[21], state[22], state[23]]
        self._rightShoulderRot = [state[24], state[25], state[26], state[27]]
        self._rightElbowRot = [state[28]]
        self._leftHipRot = [state[29], state[30], state[31], state[32]]
        self._leftKneeRot = [state[33]]
        self._leftAnkleRot = [state[34], state[35], state[36], state[37]]
        self._leftShoulderRot = [state[38], state[39], state[40], state[41]]
        self._leftElbowRot = [state[42]]

        self._baseLinVel = [state[43], state[44], state[45]]
        self._baseAngVel = [state[46], state[47], state[48]]
        self._chestVel = [state[49], state[50], state[51]]
        self._neckVel = [state[52], state[53], state[54]]
        self._rightHipVel = [state[55], state[56], state[57]]
        self._rightKneeVel = [state[58]]
        self._rightAnkleVel = [state[59], state[60], state[61]]
        self._rightShoulderVel = [state[62], state[63], state[64]]
        self._rightElbowVel = [state[65]]
        self._leftHipVel = [state[66], state[67], state[68]]
        self._leftKneeVel = [state[69]]
        self._leftAnkleVel = [state[70], state[71], state[72]]
        self._leftShoulderVel = [state[73], state[74], state[75]]
        self._leftElbowVel =[state[76]]

        pose = self.GetPose()
        return pose

    def GetPose(self):
        pose = [
            self._basePos[0], self._basePos[1], self._basePos[2], self._baseOrn[0], self._baseOrn[1],
            self._baseOrn[2], self._baseOrn[3], self._chestRot[0], self._chestRot[1],
            self._chestRot[2], self._chestRot[3], self._neckRot[0], self._neckRot[1], self._neckRot[2],
            self._neckRot[3], self._rightHipRot[0], self._rightHipRot[1], self._rightHipRot[2],
            self._rightHipRot[3], self._rightKneeRot[0], self._rightAnkleRot[0],
            self._rightAnkleRot[1], self._rightAnkleRot[2], self._rightAnkleRot[3],
            self._rightShoulderRot[0], self._rightShoulderRot[1], self._rightShoulderRot[2],
            self._rightShoulderRot[3], self._rightElbowRot[0], self._leftHipRot[0],
            self._leftHipRot[1], self._leftHipRot[2], self._leftHipRot[3], self._leftKneeRot[0],
            self._leftAnkleRot[0], self._leftAnkleRot[1], self._leftAnkleRot[2], self._leftAnkleRot[3],
            self._leftShoulderRot[0], self._leftShoulderRot[1], self._leftShoulderRot[2],
            self._leftShoulderRot[3], self._leftElbowRot[0]
        ]
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

    def getState(self):

        stateVector = []
        self.pb2dmJoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        sim_basePos, sim_baseOrn = self._p.getBasePositionAndOrientation(self.object_id[0]) 
        sim_baseLinVel, sim_baseAngVel = self._p.getBaseVelocity(self.object_id[0])
        sim_jointStates = self._p.getJointStatesMultiDof(self.object_id[0], self.pb2dmJoints)
        
        self.body_xyz = sim_basePos

        simulatedState = State(
            # Position
            basePos = sim_basePos,
            baseOrn = sim_baseOrn,
            chestRot = sim_jointStates[chest][0], 
            neckRot = sim_jointStates[neck][0], 
            rightHipRot = sim_jointStates[rightHip][0], 
            rightKneeRot = sim_jointStates[rightKnee][0], 
            rightAnkleRot = sim_jointStates[rightAnkle][0], 
            rightShoulderRot = sim_jointStates[rightShoulder][0], 
            rightElbowRot = sim_jointStates[rightElbow][0], 
            leftHipRot = sim_jointStates[leftHip][0], 
            leftKneeRot = sim_jointStates[leftKnee][0], 
            leftAnkleRot = sim_jointStates[leftAnkle][0], 
            leftShoulderRot = sim_jointStates[leftShoulder][0], 
            leftElbowRot = sim_jointStates[leftElbow][0],

            # Velocity
            baseLinVel = sim_baseLinVel,
            baseAngVel = sim_baseAngVel,
            chestVel = sim_jointStates[chest][1], 
            neckVel = sim_jointStates[neck][1], 
            rightHipVel = sim_jointStates[rightHip][1], 
            rightKneeVel = sim_jointStates[rightKnee][1], 
            rightAnkleVel = sim_jointStates[rightAnkle][1], 
            rightShoulderVel = sim_jointStates[rightShoulder][1], 
            rightElbowVel = sim_jointStates[rightElbow][1], 
            leftHipVel = sim_jointStates[leftHip][1], 
            leftKneeVel = sim_jointStates[leftKnee][1], 
            leftAnkleVel = sim_jointStates[leftAnkle][1], 
            leftShoulderVel = sim_jointStates[leftShoulder][1], 
            leftElbowVel = sim_jointStates[leftElbow][1],
        )
        simulatedState = self.get_state_array(simulatedState).tolist()
        return simulatedState

    def get_state_array(self, state):
        pos = list(state.basePos) + list(state.baseOrn) + list(state.chestRot) +list(state.neckRot) + list(state.rightHipRot)+ list(state.rightKneeRot) +list(state.rightAnkleRot) +list(state.rightShoulderRot) +list(state.rightElbowRot) +list(state.leftHipRot) +list(state.leftKneeRot) +list(state.leftAnkleRot) +list(state.leftShoulderRot) +list(state.leftElbowRot)
        
        vel = list(state.baseLinVel) + list(state.baseAngVel) + list(state.chestVel) +list(state.neckVel) + list(state.rightHipVel)+ list(state.rightKneeVel) +list(state.rightAnkleVel) +list(state.rightShoulderVel) +list(state.rightElbowVel) +list(state.leftHipVel) +list(state.leftKneeVel) +list(state.leftAnkleVel) +list(state.leftShoulderVel) +list(state.leftElbowVel)

        return np.array(pos + vel)

    def reset(self, _mocap_data, _frame, _frameNext, frameFraction, _cycleCount, mocap=True):
        
        if mocap:
            pose = self.computePose(_mocap_data, _frame, _frameNext, frameFraction, _cycleCount)
            self.initializePose(self.object_id[0], initBase=True, initializeVelocity=True)
            self.initializePose(self._kin_model, initBase=True, initializeVelocity=True)
        else:
            self.base_joint_angles = self.ConvertFromAction(_mocap_data)
            self.initializePose(self.object_id[0], initBase=True)
            # self.initializePose(self._kin_model, initBase=True, initializeVelocity=False)

        # init_action = np.zeros(self.action_dim)
        # self.base_joint_angles = self.ConvertFromAction(init_action)
        
        # useArray = True
        # self._p.resetBasePositionAndOrientation(self.object_id[0], self._basePos,
        #                                                     self._baseOrn)
        # self._p.resetBaseVelocity(self.object_id[0], self._baseLinVel, self._baseAngVel)
        # if useArray:
        #     indices = [chest,neck,rightHip,rightKnee,
        #             rightAnkle, rightShoulder, rightElbow,leftHip,
        #             leftKnee, leftAnkle, leftShoulder,leftElbow]
        #     jointPositions = [self._chestRot, self._neckRot, self._rightHipRot, self._rightKneeRot,
        #                     self._rightAnkleRot, self._rightShoulderRot, self._rightElbowRot, self._leftHipRot,
        #                     self._leftKneeRot, self._leftAnkleRot, self._leftShoulderRot, self._leftElbowRot]
            
        #     jointVelocities = [self._chestVel, self._neckVel, self._rightHipVel, self._rightKneeVel,
        #                     self._rightAnkleVel, self._rightShoulderVel, self._rightElbowVel, self._leftHipVel,
        #                     self._leftKneeVel, self._leftAnkleVel, self._leftShoulderVel, self._leftElbowVel]
        #     self._p.resetJointStatesMultiDof(self.object_id[0], indices,
        #                                                 jointPositions, jointVelocities)
        # else:
        #     self._p.resetJointStateMultiDof(self.object_id[0], chest, self._chestRot,
        #                                                 self._chestVel)
        #     self._p.resetJointStateMultiDof(self.object_id[0], neck, self._neckRot, self._neckVel)
        #     self._p.resetJointStateMultiDof(self.object_id[0], rightHip, self._rightHipRot,
        #                                                 self._rightHipVel)
        #     self._p.resetJointStateMultiDof(self.object_id[0], rightKnee, self._rightKneeRot,
        #                                                 self._rightKneeVel)
        #     self._p.resetJointStateMultiDof(self.object_id[0], rightAnkle, self._rightAnkleRot,
        #                                                 self._rightAnkleVel)
        #     self._p.resetJointStateMultiDof(self.object_id[0], rightShoulder,
        #                                                 self._rightShoulderRot, self._rightShoulderVel)
        #     self._p.resetJointStateMultiDof(self.object_id[0], rightElbow, self._rightElbowRot,
        #                                                 self._rightElbowVel)
        #     self._p.resetJointStateMultiDof(self.object_id[0], leftHip, self._leftHipRot,
        #                                                 self._leftHipVel)
        #     self._p.resetJointStateMultiDof(self.object_id[0], leftKnee, self._leftKneeRot,
        #                                                 self._leftKneeVel)
        #     self._p.resetJointStateMultiDof(self.object_id[0], leftAnkle, self._leftAnkleRot,
        #                                                 self._leftAnkleVel)
        #     self._p.resetJointStateMultiDof(self.object_id[0], leftShoulder,
        #                                                 self._leftShoulderRot, self._leftShoulderVel)
        #     self._p.resetJointStateMultiDof(self.object_id[0], leftElbow, self._leftElbowRot,
        #                                                 self._leftElbowVel)

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
            desiredVelocities = [0] * 43

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
        angDiff = [0, 0, 0]  #self.computeAngVel(curOrn, desiredOrn, 1, self._p)
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

    def apply_action(self, a):
        
        self.desiredPose = self.ConvertFromAction(a)
        #we need the target root positon and orientation to be zero, to be compatible with deep mimic
        self.desiredPose[0] = 0
        self.desiredPose[1] = 0
        self.desiredPose[2] = 0
        self.desiredPose[3] = 0
        self.desiredPose[4] = 0
        self.desiredPose[5] = 0
        self.desiredPose[6] = 0
        self.desiredPose = np.array(self.desiredPose)

        self.desiredPose = self.desiredPose #+ np.array(kinpose)

        taus = self.computePDForces(self.desiredPose,
                                            desiredVelocities=None,
                                            maxForces=self.maxForces)
        #taus = [0]*43
        self.applyPDForces(taus)

        # self.computeAndApplyPDForces(self.desiredPose, maxForces=self.maxForces)

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

    def initializePose(self, phys_model, initBase, initializeVelocity=True):
        
        useArray = True
        if initializeVelocity:
            if initBase:
                self._p.resetBasePositionAndOrientation(phys_model, self._basePos,
                                                                    self._baseOrn)
                self._p.resetBaseVelocity(phys_model, self._baseLinVel, self._baseAngVel)
            if useArray:
                indices = [chest,neck,rightHip,rightKnee,
                        rightAnkle, rightShoulder, rightElbow,leftHip,
                        leftKnee, leftAnkle, leftShoulder,leftElbow]
                jointPositions = [self._chestRot, self._neckRot, self._rightHipRot, self._rightKneeRot,
                                self._rightAnkleRot, self._rightShoulderRot, self._rightElbowRot, self._leftHipRot,
                                self._leftKneeRot, self._leftAnkleRot, self._leftShoulderRot, self._leftElbowRot]

                jointVelocities = [self._chestVel, self._neckVel, self._rightHipVel, self._rightKneeVel,
                                 self._rightAnkleVel, self._rightShoulderVel, self._rightElbowVel, self._leftHipVel,
                                self._leftKneeVel, self._leftAnkleVel, self._leftShoulderVel, self._leftElbowVel]
                self._p.resetJointStatesMultiDof(phys_model, indices,
                                                            jointPositions, jointVelocities)
            else:
                self._p.resetJointStateMultiDof(phys_model, chest, self._chestRot,
                                                            self._chestVel)
                self._p.resetJointStateMultiDof(phys_model, neck, self._neckRot, self._neckVel)
                self._p.resetJointStateMultiDof(phys_model, rightHip, self._rightHipRot,
                                                            self._rightHipVel)
                self._p.resetJointStateMultiDof(phys_model, rightKnee, self._rightKneeRot,
                                                            self._rightKneeVel)
                self._p.resetJointStateMultiDof(phys_model, rightAnkle, self._rightAnkleRot,
                                                            self._rightAnkleVel)
                self._p.resetJointStateMultiDof(phys_model, rightShoulder,
                                                            self._rightShoulderRot, self._rightShoulderVel)
                self._p.resetJointStateMultiDof(phys_model, rightElbow, self._rightElbowRot,
                                                            self._rightElbowVel)
                self._p.resetJointStateMultiDof(phys_model, leftHip, self._leftHipRot,
                                                            self._leftHipVel)
                self._p.resetJointStateMultiDof(phys_model, leftKnee, self._leftKneeRot,
                                                            self._leftKneeVel)
                self._p.resetJointStateMultiDof(phys_model, leftAnkle, self._leftAnkleRot,
                                                            self._leftAnkleVel)
                self._p.resetJointStateMultiDof(phys_model, leftShoulder,
                                                            self._leftShoulderRot, self._leftShoulderVel)
                self._p.resetJointStateMultiDof(phys_model, leftElbow, self._leftElbowRot,
                                                            self._leftElbowVel)
        else:
        
            if initBase:
                self._p.resetBasePositionAndOrientation(phys_model, self._basePos,
                                                                    self._baseOrn)
            if useArray:
                indices = [chest,neck,rightHip,rightKnee,
                        rightAnkle, rightShoulder, rightElbow,leftHip,
                        leftKnee, leftAnkle, leftShoulder,leftElbow]
                jointPositions = [self._chestRot, self._neckRot, self._rightHipRot, self._rightKneeRot,
                                self._rightAnkleRot, self._rightShoulderRot, self._rightElbowRot, self._leftHipRot,
                                self._leftKneeRot, self._leftAnkleRot, self._leftShoulderRot, self._leftElbowRot]
                self._p.resetJointStatesMultiDof(phys_model, indices,jointPositions)
                
            else:
                self._p.resetJointStateMultiDof(phys_model, chest, self._chestRot, [0, 0, 0])
                self._p.resetJointStateMultiDof(phys_model, neck, self._neckRot, [0, 0, 0])
                self._p.resetJointStateMultiDof(phys_model, rightHip, self._rightHipRot,
                                                            [0, 0, 0])
                self._p.resetJointStateMultiDof(phys_model, rightKnee, self._rightKneeRot, [0])
                self._p.resetJointStateMultiDof(phys_model, rightAnkle, self._rightAnkleRot,
                                                            [0, 0, 0])
                self._p.resetJointStateMultiDof(phys_model, rightShoulder,
                                                            self._rightShoulderRot, [0, 0, 0])
                self._p.resetJointStateMultiDof(phys_model, rightElbow, self._rightElbowRot,
                                                            [0])
                self._p.resetJointStateMultiDof(phys_model, leftHip, self._leftHipRot,
                                                            [0, 0, 0])
                self._p.resetJointStateMultiDof(phys_model, leftKnee, self._leftKneeRot, [0])
                self._p.resetJointStateMultiDof(phys_model, leftAnkle, self._leftAnkleRot,
                                                            [0, 0, 0])
                self._p.resetJointStateMultiDof(phys_model, leftShoulder,
                                                            self._leftShoulderRot, [0, 0, 0])
                self._p.resetJointStateMultiDof(phys_model, leftElbow, self._leftElbowRot, [0])
