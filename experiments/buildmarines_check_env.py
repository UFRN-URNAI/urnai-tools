
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gymnasium import spaces
from pysc2.env import sc2_env
from stable_baselines3.common.env_checker import check_env

from urnai.environments.stablebaselines3.custom_env import CustomEnv
from urnai.sc2.actions.buildmarines import BuildMarinesActionSpace
from urnai.sc2.environments.sc2environment import SC2Env
from urnai.sc2.rewards.buildmarines import BuildMarinesReward
from urnai.sc2.states.buildmarines import BuildMarinesState

players = [sc2_env.Agent(sc2_env.Race.terran)]
action_space = spaces.Discrete(n=4, start=0)
observation_space = spaces.Box(low=0, high=255, shape=(4, ), dtype=float)

env = SC2Env(map_name='BuildMarines', visualize=False, 
            step_mul=16, players=players)
state = BuildMarinesState()
urnai_action_space = BuildMarinesActionSpace()
reward = BuildMarinesReward()
custom_env = CustomEnv(env, state, urnai_action_space, reward, observation_space, 
                    action_space)

check_env(custom_env, warn=True)