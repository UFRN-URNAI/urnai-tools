"""This file is a repository with reward classes for all gym games we've solved."""
from urnai.agents.rewards.abreward import RewardBuilder


class FrozenlakeReward(RewardBuilder):
    def get_reward(self, obs, reward, done):
        if reward == 1:
            return 1000
        elif reward == 0:
            return 1
        else:
            return -1000


class FrozenlakeJiexunseeReward(RewardBuilder):
    def get_reward(self, obs, reward, done):
        if reward == 0:
            reward = -0.01
        if done:
            if reward < 1:
                reward = -1
        return reward
