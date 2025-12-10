from pysc2.env import sc2_env

from urnai.sc2.actions.collectables import CollectablesActionSpace
from urnai.sc2.environments.sc2environment import SC2Env
from urnai.sc2.rewards.collectables import CollectablesReward
from urnai.sc2.states.deepmind_state import DeepmindState

class QuickEnv:
    def __init__(self):
        players = [sc2_env.Agent(sc2_env.Race.terran)]
        self._env = SC2Env(map_name='CollectMineralShards',
                            visualize=True,
                            step_mul=16,
                            players=players)

        self._state = DeepmindState()
        self._action_space = CollectablesActionSpace()
        self._reward = CollectablesReward()

        self._obs = self._env.reset()
    
    def step(self, action):
        action = self._action_space.get_action(action, self._obs)

        obs, reward, terminated, truncated = self._env.step(action)

        self._obs = obs
        obs = self._state.update(self._obs)
        reward = self._reward.get(self._obs, reward, terminated, truncated)
        return obs, reward, terminated or truncated
    
    def reset(self):
        self._obs = self._env.reset()
        self._reward.reset()
        self._action_space.reset()
        self._state.reset()