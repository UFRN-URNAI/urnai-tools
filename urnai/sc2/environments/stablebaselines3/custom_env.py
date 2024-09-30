import gymnasium as gym
from gymnasium import spaces

from urnai.actions.action_space_base import ActionSpaceBase
from urnai.environments.environment_base import EnvironmentBase
from urnai.rewards.reward_base import RewardBase
from urnai.states.state_base import StateBase


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, env: EnvironmentBase, state: StateBase, 
                 urnai_action_space: ActionSpaceBase, reward: RewardBase, 
                 observation_space: spaces.Space, action_space: spaces.Space):
        super().__init__()

        self._env = env
        self._state = state
        self._action_space = urnai_action_space
        self._reward = reward
        self._obs = None
        # SB3 spaces
        self.action_space = action_space
        self.observation_space = observation_space

    def step(self, action):
        action = self._action_space.get_action(action, self._obs)

        obs, reward, terminated, truncated = self._env.step(action)

        self._obs = obs[0]
        obs = self._state.update(self._obs)
        reward = self._reward.get_reward(self._obs, reward[0], terminated, truncated)
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs = self._env.reset()
        self._obs = obs[0]
        obs = self._state.update(self._obs)
        info = {}
        return obs, info

    def render(self):
        pass

    def close(self):
        self._env.close()