from typing import Union

import numpy as np
from pysc2.lib import units
from stable_baselines3.common.type_aliases import GymResetReturn, GymStepReturn

import urnai.sc2.actions.sc2_actions_aux as sc2aux
from urnai.environments.stablebaselines3.custom_env import CustomEnv

EPISODE_MINUTES = 5
STEPS_PER_SECOND = 22
STEPS_PER_MINUTE = int(STEPS_PER_SECOND * 60)

class CustomEnvBuildMarines(CustomEnv):
    """Custom Environment for Build Marines that follows gym interface."""
    
    def __init__(self, env, state, urnai_action_space, reward, observation_space, 
                 action_space, logger, step_mul=32, 
                 max_steps= EPISODE_MINUTES * STEPS_PER_MINUTE):
        super().__init__(env, state, urnai_action_space, reward, observation_space, 
                         action_space)
        self.actions = {0: "Collect", 1: "BuildSupplyDepot",
                        2: "BuildBarrack", 3: "BuildMarine"}
        self.action_map_count = {"Collect": 0, "BuildSupplyDepot": 0, 
                                 "BuildBarrack": 0, "BuildMarine": 0}
        self.action_map_reward = {"Collect": 0, "BuildSupplyDepot": 0, 
                                 "BuildBarrack": 0, "BuildMarine": 0}
        self.logger = logger
        self.max_steps = max_steps
        self.step_count = 0
        self.step_mul = step_mul

        self.ep_reward = 0
    
    def step(
            self, action_idx: Union[int, np.ndarray]
        ) -> GymStepReturn:
        action, action_info = self._action_space.get_action(action_idx, self._obs)

        obs, reward, terminated, truncated = self._env.step(action)

        self._obs = obs
        obs = self._state.update(self._obs, action_idx)
        reward = self._reward.get(self._obs, reward, terminated, truncated, action_info=action_info)
        info = {}

        action_name = self.actions[action_idx]
        self.action_map_reward[action_name] += reward
        self.action_map_count[action_name] += 1
        self.step_count += self.step_mul

        if self.step_count >= self.max_steps:
            print("Max steps reached.")
            truncated = True

        self.ep_reward += reward
        reward = 0

        if terminated or truncated:
            reward = self.ep_reward
            self.log_reward_per_action()

        return obs, reward, terminated, truncated, info
    
    def reset(
            self, seed: int = None, options: dict = None
        ) -> GymResetReturn:
        self.action_map_count = {action: 0 for action in self.actions.values()}
        self.action_map_reward = {action: 0 for action in self.actions.values()}
        self.step_count = 0
        self.ep_reward = 0
        return super().reset(seed=seed, options=options)

    def log_reward_per_action(self):
        print("Episode finished.")
        log_dict = {}
        for action, count in self.action_map_count.items():
            avg_reward = 0.00
            if count > 0:
                avg_reward = self.action_map_reward[action] / count
            log_dict[f"action/count/{action}"] = count
            log_dict[f"action/avg_reward/{action}"] = avg_reward
            print(f"Action: {action}, Action Count: {count}, \
                  Average Reward: {avg_reward:.2f}")
        print("Total Reward: ", self._reward.total_reward)
        log_dict["total_reward"] = self._reward.total_reward
        marines = sc2aux.get_my_units_amount(self._obs, units.Terran.Marine)
        print("Marines Built: ", marines)
        log_dict["marines_built"] = marines
        supply_depots = sc2aux.get_my_units_amount(self._obs, units.Terran.SupplyDepot)
        # print("Supply Depots Built: ", supply_depots)
        log_dict["supply_depots_built"] = supply_depots
        barracks = sc2aux.get_my_units_amount(self._obs, units.Terran.Barracks)
        # print("Barracks Built: ", barracks)
        log_dict["barracks_built"] = barracks
        # print("Supply Depots Built: ", self._reward.supply_depots_built)
        # print("Barracks Built: ", self._reward.barracks_built)
        if self.logger:
            self.logger.log(log_dict)

    def get_action_mask(self) -> np.ndarray:
        """Get the action mask for the current observation."""
        excluded_actions_idx = self._action_space.get_excluded_actions(self._obs)
        mask = np.ones(len(self.actions), dtype=bool)
        for idx in excluded_actions_idx:
            mask[idx] = False
        return mask


        