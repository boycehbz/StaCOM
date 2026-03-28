'''
 @FileName    : env_base.py
 @EditTime    : 2021-12-20 14:59:54
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''


import gym
import gym.utils.seeding
import numpy as np
import pybullet
import os
import sys
sys.path.append('./')
import pybullet_data
import math
from vclrl_envs.bullet_utils import BulletClient, Camera


class EnvBase(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}
    _render_width = 320 * 3
    _render_height = 240 * 3

    def __init__(self, character_class, render=False, data=None, num_agents=2, **kwargs):
        self.data_path = data
        self.character_path = os.path.join(self.data_path, 'character.urdf')
        self.character_offset = os.path.join(self.data_path, 'trans_offset.txt')
        self.scene_path = os.path.join(self.data_path, 'scene.urdf')
        self.character_class = character_class

        self.num_agents = int(num_agents)
        self.is_render = render

        self.seed()
        self.load_engine()
        self.load_scene()
        self.load_character()

    def load_character(self):
        self.robots = []
        for _ in range(self.num_agents):
            r = self.character_class(self._p, self.simulation_step)
            r.load_robot_model(self.character_path, self.character_offset, self.lateralFriction)
            # r.pose_base_init()
            self.robots.append(r)

        self.state_id = self._p.saveState()

    def load_engine(self):
        bc_mode = pybullet.GUI if self.is_render else pybullet.DIRECT
        self._p = BulletClient(connection_mode=bc_mode)

        if self.is_render:
            self.camera = Camera(self._p, 1 / self.control_step)
            if hasattr(self, "create_target"):
                self.create_target()

        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        self._p.setPhysicsEngineParameter(
            numSolverIterations=10,
            numSubSteps=8, #2,
            deterministicOverlappingPairs=1
        )
        self._p.setGravity(0, -self.gravity, 0)
        self._p.setTimeStep(self.simulation_step)

    def load_scene(self):
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if os.path.exists(self.scene_path):
            z2y = self._p.getQuaternionFromEuler([0., 0, 0]) 
            self._planeId = self._p.loadURDF(self.scene_path, [0, 0, 0], z2y, useMaximalCoordinates=True, globalScaling=1.)
        else:
            z2y = self._p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0]) 
            self._planeId = self._p.loadURDF(R"plane_implicit.urdf", [0, -0.05, 0], z2y, useMaximalCoordinates=True)
            
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_Y_AXIS_UP, 1)
        self._p.changeDynamics(self._planeId, linkIndex=-1, lateralFriction=self.lateralFriction)

        self._p.changeDynamics(self._planeId, linkIndex=-1, spinningFriction=200000, rollingFriction=200000, contactStiffness=200000, contactDamping=20000)

        self.ground_ids = self._planeId

    def close(self):
        if self.owns_physics_client and self.physics_client_id >= 0:
            self._p.disconnect()
        self.physics_client_id = -1


    def set_env_params(self, params_dict):
        for k, v in params_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def set_robot_params(self, params_dict):
        for r in self.robots:
            for k, v in params_dict.items():
                if hasattr(r, k):
                    setattr(r, k, v)
            r.calc_torque_limits()

    def render(self, mode="human"):
        # Taken care of by pybullet
        if not self.is_render:
            self.is_render = True
            self._p.disconnect()
            self.initialize_scene_and_robot()
            self.reset()

        if mode != "rgb_array":
            return np.array([])

        yaw, pitch, dist, lookat = self._p.getDebugVisualizerCamera()[-4:]

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=lookat,
            distance=dist,
            yaw=yaw,
            pitch=pitch,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self._render_width) / self._render_height,
            nearVal=0.1,
            farVal=100.0,
        )
        (_, _, px, _, _) = self._p.getCameraImage(
            width=self._render_width,
            height=self._render_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb_array = np.array(px)
        rgb_array = np.reshape(
            np.array(px), (self._render_height, self._render_width, -1)
        )
        rgb_array = rgb_array[:, :, :3]
        return rgb_array.astype(np.uint8)

    def reset(self):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, a):
        raise NotImplementedError

    def _handle_keyboard(self):
        keys = self._p.getKeyboardEvents()
        # keys is a dict, so need to check key exists
        if ord("d") in keys and keys[ord("d")] == self._p.KEY_WAS_RELEASED:
            self.debug = True if not hasattr(self, "debug") else not self.debug
        elif ord("r") in keys and keys[ord("r")] == self._p.KEY_WAS_RELEASED:
            self.done = True
        elif ord("z") in keys and keys[ord("z")] == self._p.KEY_WAS_RELEASED:
            self._p.configureDebugVisualizer(
                self._p.COV_ENABLE_SINGLE_STEP_RENDERING, 0
            )
            while True:
                keys = self._p.getKeyboardEvents()
                if ord("z") in keys and keys[ord("z")] == self._p.KEY_WAS_RELEASED:
                    break
