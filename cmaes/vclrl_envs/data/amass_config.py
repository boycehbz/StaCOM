'''
 @FileName    : amass_config.py
 @EditTime    : 2021-12-20 16:53:11
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : Configuration parameters for characters in AMASS strcture.
'''

import numpy as np

root = -1
lhip = 0
lknee = 1
lankle = 2
rhip = 3
rknee = 4
rankle = 5
lowerback = 6
upperback = 7
chest = 8
lowerneck = 9
upperneck = 10
lclavicle = 11
lshoulder = 12
lelbow = 13
lwrist = 14     # fixed
rclavicle = 15
rshoulder = 16
relbow = 17
rwrist = 18     # fixed

jointFrictionForce = 0

joint_names = [
    'trans_x',
    'trans_y',
    'trans_z',
    'rot_x',
    'rot_y',
    'rot_z',
    'lhip_x',
    'lhip_y',
    'lhip_z',
    'lknee_x',
    'lknee_y',
    'lknee_z',
    'lankle_x',
    'lankle_y',
    'lankle_z',
    'rhip_x',
    'rhip_y',
    'rhip_z',
    'rknee_x',
    'rknee_y',
    'rknee_z',
    'rankle_x',
    'rankle_y',
    'rankle_z',
    'lback_x',
    'lback_y',
    'lback_z',
    'uback_x',
    'uback_y',
    'uback_z',
    'chest_x',
    'chest_y',
    'chest_z',
    'lneck_x',
    'lneck_y',
    'lneck_z',
    'uneck_x',
    'uneck_y',
    'uneck_z',
    'lclavicle_x',
    'lclavicle_y',
    'lclavicle_z',
    'lshoulder_x',
    'lshoulder_y',
    'lshoulder_z',
    'lelbow_x',
    'lelbow_y',
    'lelbow_z',
    'rclavicle_x',
    'rclavicle_y',
    'rclavicle_z',
    'rshoulder_x',
    'rshoulder_y',
    'rshoulder_z',
    'relbow_x',
    'relbow_y',
    'relbow_z',

    'vtrans_x',
    'vtrans_y',
    'vtrans_z',
    'vrot_x',
    'vrot_y',
    'vrot_z',
    'vlhip_x',
    'vlhip_y',
    'vlhip_z',
    'vlknee_x',
    'vlknee_y',
    'vlknee_z',
    'vlankle_x',
    'vlankle_y',
    'vlankle_z',
    'vrhip_x',
    'vrhip_y',
    'vrhip_z',
    'vrknee_x',
    'vrknee_y',
    'vrknee_z',
    'vrankle_x',
    'vrankle_y',
    'vrankle_z',
    'vlback_x',
    'vlback_y',
    'vlback_z',
    'vuback_x',
    'vuback_y',
    'vuback_z',
    'vchest_x',
    'vchest_y',
    'vchest_z',
    'vlneck_x',
    'vlneck_y',
    'vlneck_z',
    'vuneck_x',
    'vuneck_y',
    'vuneck_z',
    'vlclavicle_x',
    'vlclavicle_y',
    'vlclavicle_z',
    'vlshoulder_x',
    'vlshoulder_y',
    'vlshoulder_z',
    'vlelbow_x',
    'vlelbow_y',
    'vlelbow_z',
    'vrclavicle_x',
    'vrclavicle_y',
    'vrclavicle_z',
    'vrshoulder_x',
    'vrshoulder_y',
    'vrshoulder_z',
    'vrelbow_x',
    'vrelbow_y',
    'vrelbow_z',
]

P = np.array([
                    500, 500, 500,
                    500, 500, 500, 500,

                    500, 500, 500, 500,     # left Hip
                    400, 400, 400, 400,     # left Knee
                    300, 300, 300, 300,     # left Ankle
                    500, 500, 500, 500,     # right Hip
                    400, 400, 400, 400,     # right Knee
                    300, 300, 300, 300,     # right Ankle
                    500, 500, 500, 500,     # lower back
                    500, 500, 500, 500,     # upper back
                    500, 500, 500, 500,     # chest
                    200, 200, 200, 200,     # lower neck
                    200, 200, 200, 200,     # upper neck
                    400, 400, 400, 400,     # left clavicle
                    400, 400, 400, 400,     # left Shoulder
                    300, 300, 300, 300,     # left Elbow
                    400, 400, 400, 400,     # right clavicle
                    400, 400, 400, 400,     # right Shoulder
                    300, 300, 300, 300      # right Elbow
                ])
D = P / 10.0

maxForces = [
                    500, 500, 500,
                    500, 500, 500, 500,

                    300, 300, 300, 300,     # left Hip
                    200, 200, 200, 200,     # left Knee
                    100, 100, 100, 100,     # left Ankle
                    300, 300, 300, 300,     # right Hip
                    200, 200, 200, 200,     # right Knee
                    100, 100, 100, 100,     # right Ankle
                    300, 300, 300, 300,     # lower back
                    300, 300, 300, 300,     # upper back
                    300, 300, 300, 300,     # chest
                    100, 100, 100, 100,     # lower neck
                    100, 100, 100, 100,     # upper neck
                    200, 200, 200, 200,     # left clavicle
                    200, 200, 200, 200,     # left Shoulder
                    150, 150, 150, 150,     # left Elbow
                    200, 200, 200, 200,     # right clavicle
                    200, 200, 200, 200,     # right Shoulder
                    150, 150, 150, 150      # right Elbow
                ]


