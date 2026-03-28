'''
 @FileName    : __init__.py
 @EditTime    : 2021-09-15 16:11:44
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import gym
import os


def register(id, **kvargs):
    if id in gym.envs.registration.registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, **kvargs)


# fixing package path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

register(
    id="HumanoidStandEnv-v0",
    entry_point="vclrl_envs.humanoid.HumanoidStandEnv:HumanoidStandEnv",
    max_episode_steps=1000,
)


register(
    id="HumanoidWalkerEnv-v0",
    entry_point="vclrl_envs.humanoid.HumanoidWalkerEnv:HumanoidWalkerEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidSampleEnv-v0",
    entry_point="vclrl_envs.humanoid.HumanoidSampleEnv:HumanoidSampleEnv",
    max_episode_steps=100000000000000,
)

register(
    id="HumanoidAdaptEnv-v0",
    entry_point="vclrl_envs.humanoid.HumanoidAdaptEnv:HumanoidAdaptEnv",
    max_episode_steps=100000000000000,
)

register(
    id="AMASSSampleEnv-v0",
    entry_point="vclrl_envs.humanoid.AMASSSampleEnv:AMASSSampleEnv",
    max_episode_steps=100000000000000,
)