import unittest
from abc import ABCMeta

from urnai.rewards.reward_base import RewardBase


class TestRewardBase(unittest.TestCase):

    def test_reset_method(self):
        # GIVEN
        RewardBase.__abstractmethods__ = set()

        class FakeReward(RewardBase):
            def __init__(self):
                super().__init__()

        reward = FakeReward()

        # WHEN
        reset_return = reward.reset()

        # THEN
        assert isinstance(RewardBase, ABCMeta)
        assert reset_return is None
    
    def test_not_implemented_get_method(self):
        # GIVEN
        RewardBase.__abstractmethods__ = set()

        class FakeReward(RewardBase):
            def __init__(self):
                super().__init__()

        reward = FakeReward()

        # WHEN / THEN
        self.assertRaises(NotImplementedError, reward.get, [[]], 0, False, False)
