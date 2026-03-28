'''
 @FileName    : data_utils.py
 @EditTime    : 2021-12-22 16:20:19
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import numpy as np
import pybullet as p
from collections import namedtuple

StateVector = namedtuple('StateVector', [
    # Position
    'basePos',
    'baseOrn',

    'leftHipRot',      # 0
    'leftKneeRot',     # 1
    'leftAnkleRot',    # 2
    'rightHipRot',     # 3
    'rightKneeRot',    # 4
    'rightAnkleRot',   # 5

    'lowerBackRot',    # 6
    'upperBackRot',    # 7
    'chestRot',        # 8
    'lowerNeckRot',    # 9
    'upperNeckRot',    # 10

    'leftClavicleRot', # 11
    'leftShoulderRot', # 12
    'leftElbowRot',    # 13
    'rightClavicleRot',# 14
    'rightShoulderRot',# 15
    'rightElbowRot',   # 16

    # Velocity
    'baseLinVel',
    'baseAngVel',

    'leftHipVel', 
    'leftKneeVel', 
    'leftAnkleVel',
    'rightHipVel', 
    'rightKneeVel', 
    'rightAnkleVel',

    'lowerBackVel',
    'upperBackVel',
    'chestVel',
    'lowerNeckVel',
    'upperNeckVel',

    'leftClavicleVel',
    'leftShoulderVel', 
    'leftElbowVel',
    'rightClavicleVel',
    'rightShoulderVel', 
    'rightElbowVel', 
])

def computePose(KeyFrameDuration, frameData, frameDataNext, frameFraction, _cycleCount):
    
    frameData = AxisAnglePose2StateVector(frameData)
    frameDataNext = AxisAnglePose2StateVector(frameDataNext)

    state = Slerp(KeyFrameDuration, frameFraction, frameData, frameDataNext, p)

    return state

def AxisAngle2Quaternion(angle):
    angle = np.array(angle)
    q = p.getQuaternionFromAxisAngle(angle/(np.linalg.norm(angle) + 1e-6), np.linalg.norm(angle))
    return list(q)

def Quaternion2AxisAngle(quat):
    axis, angle = p.getAxisAngleFromQuaternion(quat)
    angVel = [axis[0] * angle, axis[1] * angle, axis[2] * angle]
    return list(angVel)

def StateVector2StateList(state):
    state = [list(state.basePos), list(state.baseOrn) \
               , list(state.leftHipRot), list(state.leftKneeRot), list(state.leftAnkleRot) \
               , list(state.rightHipRot), list(state.rightKneeRot), list(state.rightAnkleRot) \
               , list(state.lowerBackRot), list(state.upperBackRot), list(state.chestRot), list(state.lowerNeckRot), list(state.upperNeckRot) \
               , list(state.leftClavicleRot), list(state.leftShoulderRot), list(state.leftElbowRot) \
               , list(state.rightClavicleRot), list(state.rightShoulderRot), list(state.rightElbowRot) \
               , list(state.baseLinVel), list(state.baseAngVel) \
               , list(state.leftHipVel), list(state.leftKneeVel), list(state.leftAnkleVel) \
               , list(state.rightHipVel), list(state.rightKneeVel), list(state.rightAnkleVel) \
               , list(state.lowerBackVel), list(state.upperBackVel), list(state.chestVel), list(state.lowerNeckVel), list(state.upperNeckVel) \
               , list(state.leftClavicleVel), list(state.leftShoulderVel), list(state.leftElbowVel) \
               , list(state.rightClavicleVel), list(state.rightShoulderVel), list(state.rightElbowVel) ]

    return state

def StateVector2AxisAngle(state_vector):
    state_list = StateVector2StateList(state_vector)
    axisangle = []
    for i, s in enumerate(state_list):
        if i > 0 and i < 19:
            s = Quaternion2AxisAngle(s)
        axisangle.append(s)
    axisangle = np.array(axisangle).reshape(-1,)
    return axisangle

def StateList2StateVector(state_list):
    State = StateVector(
        # Position
        basePos = state_list[0],
        baseOrn = state_list[1],

        leftHipRot = state_list[2],
        leftKneeRot = state_list[3],
        leftAnkleRot = state_list[4],
        rightHipRot = state_list[5],
        rightKneeRot = state_list[6],
        rightAnkleRot = state_list[7],

        lowerBackRot = state_list[8],
        upperBackRot = state_list[9],
        chestRot = state_list[10],
        lowerNeckRot = state_list[11],
        upperNeckRot = state_list[12],

        leftClavicleRot = state_list[13],
        leftShoulderRot = state_list[14],
        leftElbowRot = state_list[15],
        rightClavicleRot = state_list[16],
        rightShoulderRot = state_list[17],
        rightElbowRot = state_list[18],

        # Velocity
        baseLinVel = state_list[19],
        baseAngVel = state_list[20],

        leftHipVel = state_list[21],
        leftKneeVel = state_list[22],
        leftAnkleVel = state_list[23],
        rightHipVel = state_list[24],
        rightKneeVel = state_list[25],
        rightAnkleVel = state_list[26],

        lowerBackVel = state_list[27],
        upperBackVel = state_list[28],
        chestVel = state_list[29],
        lowerNeckVel = state_list[30],
        upperNeckVel = state_list[31],

        leftClavicleVel = state_list[32],
        leftShoulderVel = state_list[33],
        leftElbowVel = state_list[34],
        rightClavicleVel = state_list[35],
        rightShoulderVel = state_list[36],
        rightElbowVel = state_list[37],
    )
    return State

def AxisAngleTarPose2QuaternionTarPose(pose):
    # pose = [0] * 6 + pose.tolist()
    pose = np.array(pose).reshape(-1, 3)
    assert len(pose) == 19
    tar_pose = []
    for i, p in enumerate(pose):
        if i == 0: # translation
            tar_pose += list(p)
        else:
            tar_pose += AxisAngle2Quaternion(p)

    return tar_pose

def AxisAnglePose2StateVector(pose):
    pose = np.array(pose).reshape(-1, 3)
    assert len(pose) == 19
    state_list = []
    for i, p in enumerate(pose):
        if i == 0: # translation
            state_list.append(list(p))
        else:
            state_list.append(AxisAngle2Quaternion(p))
    for i in range(19):
        state_list.append([0,0,0])
    state_vector = StateList2StateVector(state_list)
    return state_vector

def AxisAngleState2StateVector(state):
    state = np.array(state).reshape(-1, 3)
    assert len(state) == 38
    state_list = []
    for i, p in enumerate(state):
        if i > 0 and i < 19: # translation
            state_list.append(AxisAngle2Quaternion(p))
        else:
            state_list.append(list(p))
    state_vector = StateList2StateVector(state_list)
    return state_vector

def AxisAngleState2QuaternionState(pose, KeyFrameDuration):
    pose = np.array(pose).reshape(-1, 3)

    pass

def Slerp(keyFrameDuration, frameFraction, frameData, frameDataNext, bullet_client):
    ##### Base Position
    basePos1Start = frameData.basePos
    basePos1End = frameDataNext.basePos

    _basePos = [
        basePos1Start[0] + frameFraction * (basePos1End[0] - basePos1Start[0]),
        basePos1Start[1] + frameFraction * (basePos1End[1] - basePos1Start[1]),
        basePos1Start[2] + frameFraction * (basePos1End[2] - basePos1Start[2])
    ]
    _baseLinVel = ComputeLinVel(basePos1Start, basePos1End, keyFrameDuration)
    
    ##### Base Orientation
    baseOrn1Start = frameData.baseOrn
    baseOrn1Next = frameDataNext.baseOrn
    _baseOrn = bullet_client.getQuaternionSlerp(baseOrn1Start, baseOrn1Next, frameFraction)
    _baseAngVel = ComputeAngVel(baseOrn1Start, baseOrn1Next, keyFrameDuration, bullet_client)

    ##### Left Hip
    leftHipRotStart = frameData.leftHipRot
    leftHipRotEnd = frameDataNext.leftHipRot
    _leftHipRot = bullet_client.getQuaternionSlerp(leftHipRotStart, leftHipRotEnd, frameFraction)
    _leftHipVel = ComputeAngVelRel(leftHipRotStart, leftHipRotEnd, keyFrameDuration, bullet_client)

    ##### Left Knee
    leftKneeRotStart = frameData.leftKneeRot
    leftKneeRotEnd = frameDataNext.leftKneeRot
    _leftKneeRot = bullet_client.getQuaternionSlerp(leftKneeRotStart, leftKneeRotEnd, frameFraction)
    _leftKneeVel = ComputeAngVelRel(leftKneeRotStart, leftKneeRotEnd, keyFrameDuration, bullet_client)

    ##### Left Ankle
    leftAnkleRotStart = frameData.leftAnkleRot
    leftAnkleRotEnd = frameDataNext.leftAnkleRot
    _leftAnkleRot = bullet_client.getQuaternionSlerp(leftAnkleRotStart, leftAnkleRotEnd, frameFraction)
    _leftAnkleVel = ComputeAngVelRel(leftAnkleRotStart, leftAnkleRotEnd, keyFrameDuration, bullet_client)

    ##### Right Hip
    rightHipRotStart = frameData.rightHipRot
    rightHipRotEnd = frameDataNext.rightHipRot
    _rightHipRot = bullet_client.getQuaternionSlerp(rightHipRotStart, rightHipRotEnd, frameFraction)
    _rightHipVel = ComputeAngVelRel(rightHipRotStart, rightHipRotEnd, keyFrameDuration, bullet_client)

    ##### Right Knee
    rightKneeRotStart = frameData.rightKneeRot
    rightKneeRotEnd = frameDataNext.rightKneeRot
    _rightKneeRot = bullet_client.getQuaternionSlerp(rightKneeRotStart, rightKneeRotEnd, frameFraction)
    _rightKneeVel = ComputeAngVelRel(rightKneeRotStart, rightKneeRotEnd, keyFrameDuration, bullet_client)

    ##### Right Ankle
    rightAnkleRotStart = frameData.rightAnkleRot
    rightAnkleRotEnd = frameDataNext.rightAnkleRot
    _rightAnkleRot = bullet_client.getQuaternionSlerp(rightAnkleRotStart, rightAnkleRotEnd, frameFraction)
    _rightAnkleVel = ComputeAngVelRel(rightAnkleRotStart, rightAnkleRotEnd, keyFrameDuration, bullet_client)

    ##### Lower Back
    lowerBackRotStart = frameData.lowerBackRot
    lowerBackRotEnd = frameDataNext.lowerBackRot
    _lowerBackRot = bullet_client.getQuaternionSlerp(lowerBackRotStart, lowerBackRotEnd, frameFraction)
    _lowerBackVel = ComputeAngVelRel(lowerBackRotStart, lowerBackRotEnd, keyFrameDuration, bullet_client)

    ##### Upper Back
    upperBackRotStart = frameData.upperBackRot
    upperBackRotEnd = frameDataNext.upperBackRot
    _upperBackRot = bullet_client.getQuaternionSlerp(upperBackRotStart, upperBackRotEnd, frameFraction)
    _upperBackVel = ComputeAngVelRel(upperBackRotStart, upperBackRotEnd, keyFrameDuration, bullet_client)

    ##### Chest
    chestRotStart = frameData.chestRot
    chestRotEnd = frameDataNext.chestRot
    _chestRot = bullet_client.getQuaternionSlerp(chestRotStart, chestRotEnd, frameFraction)
    _chestVel = ComputeAngVelRel(chestRotStart, chestRotEnd, keyFrameDuration, bullet_client)

    ##### Lower Neck
    lowerNeckRotStart = frameData.lowerNeckRot
    lowerNeckRotEnd = frameDataNext.lowerNeckRot
    _lowerNeckRot = bullet_client.getQuaternionSlerp(lowerNeckRotStart, lowerNeckRotEnd, frameFraction)
    _lowerNeckVel = ComputeAngVelRel(lowerNeckRotStart, lowerNeckRotEnd, keyFrameDuration, bullet_client)

    ##### Upper Neck
    upperNeckRotStart = frameData.upperNeckRot
    upperNeckRotEnd = frameDataNext.upperNeckRot
    _upperNeckRot = bullet_client.getQuaternionSlerp(upperNeckRotStart, upperNeckRotEnd, frameFraction)
    _upperNeckVel = ComputeAngVelRel(upperNeckRotStart, upperNeckRotEnd, keyFrameDuration, bullet_client)

    ##### Left Clavicle
    leftClavicleRotStart = frameData.leftClavicleRot
    leftClavicleRotEnd = frameDataNext.leftClavicleRot
    _leftClavicleRot = bullet_client.getQuaternionSlerp(leftClavicleRotStart, leftClavicleRotEnd, frameFraction)
    _leftClavicleVel = ComputeAngVelRel(leftClavicleRotStart, leftClavicleRotEnd, keyFrameDuration, bullet_client)

    ##### Left Shoulder
    leftShoulderRotStart = frameData.leftShoulderRot
    leftShoulderRotEnd = frameDataNext.leftShoulderRot
    _leftShoulderRot = bullet_client.getQuaternionSlerp(leftShoulderRotStart, leftShoulderRotEnd, frameFraction)
    _leftShoulderVel = ComputeAngVelRel(leftShoulderRotStart, leftShoulderRotEnd, keyFrameDuration, bullet_client)

    ##### Left Elbow
    leftElbowRotStart = frameData.leftElbowRot
    leftElbowRotEnd = frameDataNext.leftElbowRot
    _leftElbowRot = bullet_client.getQuaternionSlerp(leftElbowRotStart, leftElbowRotEnd, frameFraction)
    _leftElbowVel = ComputeAngVelRel(leftElbowRotStart, leftElbowRotEnd, keyFrameDuration, bullet_client)

    ##### Right Clavicle
    rightClavicleRotStart = frameData.rightClavicleRot
    rightClavicleRotEnd = frameDataNext.rightClavicleRot
    _rightClavicleRot = bullet_client.getQuaternionSlerp(rightClavicleRotStart, rightClavicleRotEnd, frameFraction)
    _rightClavicleVel = ComputeAngVelRel(rightClavicleRotStart, rightClavicleRotEnd, keyFrameDuration, bullet_client)

    ##### Right Shoulder
    rightShoulderRotStart = frameData.rightShoulderRot
    rightShoulderRotEnd = frameDataNext.rightShoulderRot
    _rightShoulderRot = bullet_client.getQuaternionSlerp(rightShoulderRotStart, rightShoulderRotEnd, frameFraction)
    _rightShoulderVel = ComputeAngVelRel(rightShoulderRotStart, rightShoulderRotEnd, keyFrameDuration, bullet_client)

    ##### Right Elbow
    rightElbowRotStart = frameData.rightElbowRot
    rightElbowRotEnd = frameDataNext.rightElbowRot
    _rightElbowRot = bullet_client.getQuaternionSlerp(rightElbowRotStart, rightElbowRotEnd, frameFraction)
    _rightElbowVel = ComputeAngVelRel(rightElbowRotStart, rightElbowRotEnd, keyFrameDuration, bullet_client)

    State = StateVector(
        # Position
        basePos = _basePos,
        baseOrn = _baseOrn,

        leftHipRot = _leftHipRot,
        leftKneeRot = _leftKneeRot,
        leftAnkleRot = _leftAnkleRot,
        rightHipRot = _rightHipRot,
        rightKneeRot = _rightKneeRot,
        rightAnkleRot = _rightAnkleRot,

        lowerBackRot = _lowerBackRot,
        upperBackRot = _upperBackRot,
        chestRot = _chestRot,
        lowerNeckRot = _lowerNeckRot,
        upperNeckRot = _upperNeckRot,

        leftClavicleRot = _leftClavicleRot,
        leftShoulderRot = _leftShoulderRot,
        leftElbowRot = _leftElbowRot,
        rightClavicleRot = _rightClavicleRot,
        rightShoulderRot = _rightShoulderRot,
        rightElbowRot = _rightElbowRot,

        # Velocity
        baseLinVel = _baseLinVel,
        baseAngVel = _baseAngVel,

        leftHipVel = _leftHipVel,
        leftKneeVel = _leftKneeVel,
        leftAnkleVel = _leftAnkleVel,
        rightHipVel = _rightHipVel,
        rightKneeVel = _rightKneeVel,
        rightAnkleVel = _rightAnkleVel,

        lowerBackVel = _lowerBackVel,
        upperBackVel = _upperBackVel,
        chestVel = _chestVel,
        lowerNeckVel = _lowerNeckVel,
        upperNeckVel = _upperNeckVel,

        leftClavicleVel = _leftClavicleVel,
        leftShoulderVel = _leftShoulderVel,
        leftElbowVel = _leftElbowVel,
        rightClavicleVel = _rightClavicleVel,
        rightShoulderVel = _rightShoulderVel,
        rightElbowVel = _rightElbowVel,
    )
    return State

def ComputeLinVel(posStart, posEnd, deltaTime):
    vel = [(posEnd[0] - posStart[0]) / deltaTime, (posEnd[1] - posStart[1]) / deltaTime,
        (posEnd[2] - posStart[2]) / deltaTime]
    return vel

def ComputeAngVel(ornStart, ornEnd, deltaTime, bullet_client):
    dorn = bullet_client.getDifferenceQuaternion(ornStart, ornEnd)
    axis, angle = bullet_client.getAxisAngleFromQuaternion(dorn)
    angVel = [(axis[0] * angle) / deltaTime, (axis[1] * angle) / deltaTime,
            (axis[2] * angle) / deltaTime]
    return angVel

def ComputeAngVelRel(ornStart, ornEnd, deltaTime, bullet_client):
    ornStartConjugate = [-ornStart[0], -ornStart[1], -ornStart[2], ornStart[3]]
    pos_diff, q_diff = bullet_client.multiplyTransforms([0, 0, 0], ornStartConjugate, [0, 0, 0],
                                                        ornEnd)
    axis, angle = bullet_client.getAxisAngleFromQuaternion(q_diff)
    angVel = [(axis[0] * angle) / deltaTime, (axis[1] * angle) / deltaTime,
            (axis[2] * angle) / deltaTime]
    return angVel