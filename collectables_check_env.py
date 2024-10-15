from gymnasium import spaces
from pysc2.env import sc2_env
from stable_baselines3.common.env_checker import check_env

from urnai.sc2.actions.collectables import CollectablesActionSpace
from urnai.sc2.environments.sc2environment import SC2Env
from urnai.sc2.environments.stablebaselines3.custom_env import CustomEnv
from urnai.sc2.rewards.collectables import CollectablesReward
from urnai.sc2.states.collectables import CollectablesState

players = [sc2_env.Agent(sc2_env.Race.terran)]
env = SC2Env(map_name='CollectMineralShards', visualize=False, 
            step_mul=16, players=players)
state = CollectablesState()
urnai_action_space = CollectablesActionSpace()
reward = CollectablesReward()

# Define action and observation space
action_space = spaces.Discrete(n=4, start=0)
observation_space = spaces.Box(low=0, high=255, shape=(4096,), dtype=float)

custom_env = CustomEnv(env, state, urnai_action_space, reward, observation_space, 
                action_space)
check_env(custom_env, warn=True)