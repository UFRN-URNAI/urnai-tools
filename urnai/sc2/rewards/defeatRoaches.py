from pysc2.lib import units

import urnai.sc2.actions.sc2_actions_aux as sc2aux

from .experiments import ExperimentsReward


class DefeatRoachesReward(ExperimentsReward):

    def __init__(self):
        super().__init__()

    def get(self, obs, default_reward, terminated, truncated) -> int:
        if self.previous_state is None:
            reward = 0
        else:
            current_roach_amount = self.get_roach_amount(obs)
            previous_roach_amount = self.get_roach_amount(self.previous_state)
            
            rwdRoaches = (current_roach_amount - previous_roach_amount)

            current_marine_amount = self.get_marine_amount(obs)
            previous_marine_amount = self.get_marine_amount(self.previous_state)

            rwdMarines = (current_marine_amount - previous_marine_amount)

            reward = (rwdMarines - rwdRoaches) * 1000
        
        self.previous_state = obs
        return reward

    def get_roach_amount(self, obs):
        return len(sc2aux.get_units_by_type(obs, units.Zerg.Roach))
    
    def get_marine_amount(self, obs):
        return len(sc2aux.get_units_by_type(obs, units.Zerg.Roach))