from typing import Union

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymStepReturn

from urnai.actions.action_space_base import ActionSpaceBase
from urnai.environments.environment_base import EnvironmentBase
from urnai.environments.stablebaselines3.custom_env import CustomEnv
from urnai.logging.logger_base import LoggerBase
from urnai.rewards.reward_base import RewardBase
from urnai.states.state_base import StateBase


class CustomEnvCollectables(CustomEnv):
    """Custom Environment for Collectables that follows gym interface."""

    def __init__(self, env: EnvironmentBase, state: StateBase, 
                 urnai_action_space: ActionSpaceBase, reward: RewardBase, 
                 observation_space: spaces.Space, action_space: spaces.Space, 
                 logger: LoggerBase):
        super().__init__(env, state, urnai_action_space, reward, observation_space, 
                         action_space)
        self.logger = logger

    def step(
            self, action: Union[int, np.ndarray]
        ) -> GymStepReturn:
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated or truncated:
            self.log_results(reward)

        return obs, reward, terminated, truncated, info

    def log_results(self, final_reward : float) -> None:
        if self.logger:
            log_dict = {}
            log_dict["total_reward"] = final_reward
            log_dict["shards_collected"] = self._reward.score
            self.logger.log(log_dict)