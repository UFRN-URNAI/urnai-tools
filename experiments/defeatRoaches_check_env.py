import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gymnasium import spaces
from pysc2.env import sc2_env
from stable_baselines3.common.env_checker import check_env

from urnai.environments.stablebaselines3.custom_env import CustomEnv
from urnai.sc2.actions.defeatRoaches import DefeatRoachesActionSpace
from urnai.sc2.environments.sc2environment import SC2Env
from urnai.sc2.rewards.defeatRoaches import DefeatRoachesReward
from urnai.sc2.states.defeatRoaches import DefeatRoachesState

players = [sc2_env.Agent(sc2_env.Race.terran)]
action_space = spaces.Discrete(n = 12, start = 0)
observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

env = SC2Env(map_name='DefeatRoaches', visualize=False, 
                step_mul=16, players=players)
state = DefeatRoachesState()
urnai_action_space = DefeatRoachesActionSpace()
reward = DefeatRoachesReward()
custom_env = CustomEnv(env, state, urnai_action_space, reward, observation_space, 
                    action_space)

check_env(custom_env, warn=True)