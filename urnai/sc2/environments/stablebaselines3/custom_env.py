from typing import Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymResetReturn, GymStepReturn

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
        # space variables, used internally by the gymnasium library
        self.action_space = action_space
        self.observation_space = observation_space

    def step(
            self, action: Union[int, np.ndarray]
        ) -> GymStepReturn:
        action = self._action_space.get_action(action, self._obs)

        obs, reward, terminated, truncated = self._env.step(action)

        self._obs = obs[0]
        obs = self._state.update(self._obs)
        reward = self._reward.get(self._obs, reward[0], terminated, truncated)
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(
            self, seed: int = None, options: dict = None
        ) -> GymResetReturn:
        obs = self._env.reset()
        self._obs = obs[0]
        obs = self._state.update(self._obs)
        info = {}
        return obs, info

    def render(self, mode: str) -> None:
        raise NotImplementedError("Render method not implemented. If necessary, you " +
                                  "should implement it in your CustomEnv subclass.")

    def close(self) -> None:
        self._env.close()