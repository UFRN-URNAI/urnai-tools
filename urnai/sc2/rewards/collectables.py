import numpy as np

import urnai.sc2.actions.sc2_actions_aux as sc2aux
from urnai.rewards.reward_base import RewardBase

STATE_MAXIMUM_NUMBER_OF_MINERAL_SHARDS = 20


class CollectablesReward(RewardBase):

    def __init__(self):
        self.previous_state = None
        self.old_collectable_counter = STATE_MAXIMUM_NUMBER_OF_MINERAL_SHARDS
        self.score = 0

    def get(self, obs, default_reward, terminated, truncated) -> int:
        
        reward = 0
        if(self.previous_state is not None):
            current = self.filter_non_mineral_shard_units(obs)
            curr = np.count_nonzero(current == 1)
            if curr != self.old_collectable_counter:
                self.score += (self.old_collectable_counter - curr) \
                              % STATE_MAXIMUM_NUMBER_OF_MINERAL_SHARDS
                self.old_collectable_counter = curr
                reward = 10
            else:
                reward = -1
        
        if(truncated or terminated):

            print("Score: ", self.score)
            
            if(self.score >= 20):
                reward = 1000
            elif(self.score >= 15):
                reward = 500
            elif(self.score >= 10):
                reward = 100
            elif(self.score >= 5):
                reward = -100
            elif(self.score >= 1):
                reward = -500
            else:
                reward = -10000
        
        self.previous_state = obs
        return reward
    
    def reset(self) -> None:
        self.previous_state = None
        self.old_collectable_counter = STATE_MAXIMUM_NUMBER_OF_MINERAL_SHARDS
        self.score = 0

    def filter_non_mineral_shard_units(self, obs):
        filtered_map = np.zeros((len(obs.feature_minimap[0]), 
                                 len(obs.feature_minimap[0][0])))
        for unit in sc2aux.get_all_neutral_units(obs):
            filtered_map[unit.y][unit.x] = 1

        return filtered_map