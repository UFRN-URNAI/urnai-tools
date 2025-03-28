from abc import abstractmethod

from urnai.rewards.reward_base import RewardBase

STATE_MAXIMUM_NUMBER_OF_MINERAL_SHARDS = 20


class ExperimentsReward(RewardBase):

    def __init__(self):
        self.previous_state = None

    @abstractmethod
    def get(self, obs, default_reward, terminated, truncated) -> int:
        ...
    
    def reset(self) -> None:
        self.previous_state = None