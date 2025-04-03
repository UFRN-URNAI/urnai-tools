from pysc2.lib import units

import urnai.sc2.actions.sc2_actions_aux as sc2aux
from urnai.rewards.reward_base import RewardBase


class BuildMarinesReward(RewardBase):

    def __init__(self):
        self.previous_state = None

    def get(self, obs, default_reward, terminated, truncated) -> int:

        reward = default_reward

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
            
            reward = (supply_depot_amount_diff * 1) + (barracks_amount_diff * 5) + \
                    (marines_amount_diff * 20)

        
        self.previous_state = obs
        return reward
    
    def reset(self) -> None:
        self.previous_state = None
        self.score = 0