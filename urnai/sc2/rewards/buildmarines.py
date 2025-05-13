from pysc2.lib import units

import urnai.sc2.actions.sc2_actions_aux as sc2aux
from urnai.rewards.reward_base import RewardBase


class BuildMarinesReward(RewardBase):

    def __init__(self, config_dict: dict = None) -> None:
        self.previous_state = None
        self.actions = {0: "Collect", 1: "BuildSupplyDepot",
                        2: "BuildBarrack", 3: "BuildMarine"}
        self.action_map_count = {"Collect": 0, "BuildSupplyDepot": 0, 
                                 "BuildBarrack": 0, "BuildMarine": 0}
        self.increasing_mineral_count = 0
        self.total_reward = 0
        config_dict = config_dict or {}
        self.w_supply = config_dict.get("w_supply", 3.0)
        self.w_barrack = config_dict.get("w_barrack", 5.0)
        self.w_marine = config_dict.get("w_marine", 1.5)
        self.penalty_no_supply = config_dict.get("penalty_no_supply", 0.15)
        self.penalty_no_barrack = config_dict.get("penalty_no_barrack", 0.1)

    def get(self, obs, default_reward, terminated, truncated, action_idx = -1) -> int:

        reward = 0

        if(self.previous_state is not None):

            current_supply = sc2aux.get_my_units_amount(obs, units.Terran.SupplyDepot)
            prev_supply = sc2aux.get_my_units_amount(self.previous_state, 
                                                     units.Terran.SupplyDepot)
            supply_depot_amount_diff = max(current_supply - prev_supply, 0)

            current_barracks = sc2aux.get_my_units_amount(obs, units.Terran.Barracks)
            prev_barracks = sc2aux.get_my_units_amount(self.previous_state, 
                                                       units.Terran.Barracks)
            barracks_amount_diff = max(current_barracks - prev_barracks, 0)

            current_marines = sc2aux.get_my_units_amount(obs, units.Terran.Marine)
            prev_marines = sc2aux.get_my_units_amount(self.previous_state, 
                                                      units.Terran.Marine)
            marines_amount_diff = max(current_marines - prev_marines, 0)

            reward = (supply_depot_amount_diff * self.w_supply) + \
                        (barracks_amount_diff * self.w_barrack) + \
                        (marines_amount_diff * self.w_marine)

            # Negative reward for not having a supply depot built
            if(current_supply == 0):
                reward -= self.penalty_no_supply
            # Negative reward for not having a barrack built
            if(current_barracks == 0):
                reward -= self.penalty_no_barrack
        
        self.total_reward += reward
        self.previous_state = obs
        return reward
    
    def reset(self) -> None:
        self.previous_state = None
        self.action_map_count = {action: 0 for action in self.actions.values()}
        self.increasing_mineral_count = 0
        self.total_reward = 0