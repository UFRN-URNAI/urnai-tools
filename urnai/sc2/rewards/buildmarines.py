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
        self.w_barrack = config_dict.get("w_barrack", 8.0)
        self.w_marine = config_dict.get("w_marine", 1.5)
        self.penalty_no_supply = config_dict.get("penalty_no_supply", 0.25)
        self.penalty_no_barrack = config_dict.get("penalty_no_barrack", 0.1)

        self.tried_to_build_barracks_before_supply = False

    def get(self, obs, default_reward, terminated, truncated, **kwargs) -> int:
        action_info = kwargs.get("action_info", {})
        if len(action_info) > 0:
            action_idx, chosen_barrack = action_info["action_idx"], action_info["chosen_barrack"]
        else:
            action_idx, chosen_barrack = 0, None

        reward = 0

        if(self.previous_state is not None):

            current_supply_depot = sc2aux.get_my_units_amount(obs, units.Terran.SupplyDepot)
            prev_supply_depot = sc2aux.get_my_units_amount(self.previous_state, 
                                                     units.Terran.SupplyDepot)
            supply_depot_amount_diff = max(current_supply_depot - prev_supply_depot, 0)

            current_barracks = sc2aux.get_my_units_amount(obs, units.Terran.Barracks)
            prev_barracks = sc2aux.get_my_units_amount(self.previous_state, 
                                                       units.Terran.Barracks)
            barracks_amount_diff = max(current_barracks - prev_barracks, 0)

            current_marines = sc2aux.get_my_units_amount(obs, units.Terran.Marine)
            prev_marines = sc2aux.get_my_units_amount(self.previous_state, 
                                                      units.Terran.Marine)
            marines_amount_diff = max(current_marines - prev_marines, 0)

            current_scv = sc2aux.get_my_units_amount(obs, units.Terran.SCV)

            reward = (supply_depot_amount_diff * self.w_supply) + \
                        (barracks_amount_diff * self.w_barrack) + \
                        (marines_amount_diff * self.w_marine)

            # Negative reward for not having a supply depot built
            if(current_supply_depot == 0):
                reward -= self.penalty_no_supply
            # Negative reward for not having a barrack built
            if(current_barracks == 0):
                reward -= self.penalty_no_barrack
            
            enough_supply_depot = 15 + current_supply_depot * 8 > current_marines + current_scv
            marine_cost = 50
            barracks_cost = 150
            supply_cost = 100
            max_supply_depot = 15
            max_barracks = 8

            # Negative reward for when certain actions are prohibited

            if (action_idx == 3): #BuildMarine

                full_queue = chosen_barrack is None or sc2aux.is_building_queue_full(chosen_barrack)

                if (current_barracks == 0 or 
                    not enough_supply_depot or obs.player.minerals < marine_cost or full_queue):
                    reward -= 5

            if (action_idx == 2): #BuildBarrack
                if (prev_barracks == max_barracks or obs.player.minerals < barracks_cost):
                    reward -= 5
                if (current_supply_depot == 0):
                    reward -= 10
                    self.tried_to_build_barracks_before_supply = True

            if (action_idx == 1): #BuildSupplyDepot
                if (prev_supply_depot == max_supply_depot or obs.player.minerals < supply_cost):
                    reward -= 5
        
        self.total_reward += reward
        self.previous_state = obs
        return reward

    def reset(self) -> None:
        self.previous_state = None
        self.action_map_count = {action: 0 for action in self.actions.values()}
        self.increasing_mineral_count = 0
        self.total_reward = 0

        self.tried_to_build_barracks_before_supply = False