"""
This file should contain all reward implementations that are generic enough
to fit any agent/environment.
"""
from .abreward import RewardBuilder


class PureReward(RewardBuilder):
    def get_reward(self, obs, reward, done):
        return reward
