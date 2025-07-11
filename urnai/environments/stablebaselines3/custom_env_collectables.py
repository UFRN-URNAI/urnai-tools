from typing import Union

import numpy as np
from stable_baselines3.common.type_aliases import GymResetReturn, GymStepReturn

from urnai.environments.stablebaselines3.custom_env import CustomEnv

EPISODE_MINUTES = 2
STEPS_PER_SECOND = 16
STEPS_PER_MINUTE = int(STEPS_PER_SECOND * 60)

class CustomEnvCollectables(CustomEnv):
    """Custom Environment for Collectables that follows gym interface."""

    def __init__(self, env, state, urnai_action_space, reward, observation_space, 
                 action_space, logger, step_mul=32, 
                 max_steps= EPISODE_MINUTES * STEPS_PER_MINUTE):
        super().__init__(env, state, urnai_action_space, reward, observation_space, 
                         action_space)
        named_actions = urnai_action_space.get_named_actions()
        action_indices = urnai_action_space.get_actions()
        self.actions = {idx: name for name, idx in zip(named_actions, action_indices)}
        self.action_map_count = {name: 0 for name in named_actions}
        self.action_map_reward = {name: 0 for name in named_actions}
        self.logger = logger
        self.max_steps = max_steps
        self.step_count = 0
        self.step_mul = step_mul

    def step(
            self, action: Union[int, np.ndarray]
        ) -> GymStepReturn:
        chosen_action = self._action_space.get_action(action, self._obs)

        obs, reward, terminated, truncated = self._env.step(chosen_action)

        self.step_count += self.step_mul
        if self.step_count >= self.max_steps:
            print("Max steps reached.")
            truncated = True
        
        self._obs = obs
        obs = self._state.update(self._obs)
        reward = self._reward.get(self._obs, reward, terminated, truncated)
        info = {}

        action_name = self.actions[action]
        self.action_map_reward[action_name] += reward
        self.action_map_count[action_name] += 1

        if terminated or truncated:
            self.log_reward_per_action()

        return obs, reward, terminated, truncated, info

    def reset(
            self, seed: int = None, options: dict = None
        ) -> GymResetReturn:
        self.action_map_count = {action: 0 for action in self.actions.values()}
        self.action_map_reward = {action: 0 for action in self.actions.values()}
        self.step_count = 0
        return super().reset(seed=seed, options=options)

    def log_reward_per_action(self):
        log_dict = {}
        for action, count in self.action_map_count.items():
            avg_reward = 0.00
            if count > 0:
                avg_reward = self.action_map_reward[action] / count
            log_dict[f"action/count/{action}"] = count
            log_dict[f"action/avg_reward/{action}"] = avg_reward
        log_dict["total_reward"] = self._reward.total_reward
        log_dict["shards_collected"] = self._reward.score
        if self.logger:
            self.logger.log(log_dict)
    
    def get_action_mask(self) -> np.ndarray:
        """Get the action mask for the current observation."""
        excluded_actions_idx = self._action_space.get_excluded_actions(self._obs)
        mask = np.ones(len(self.actions), dtype=bool)
        for idx in excluded_actions_idx:
            mask[idx] = False
        return mask