import unittest
from abc import ABCMeta

from urnai.actions.action_base import ActionBase
from urnai.actions.action_space_base import ActionSpaceBase
from urnai.strategies.strategy_base import StrategyBase


class FakeStrategy(StrategyBase):
    StrategyBase.__abstractmethods__ = set()

class FakeActionSpace(ActionSpaceBase):
    ActionSpaceBase.__abstractmethods__ = set()

class TestFakeStrategy(unittest.TestCase):
    
    def test_abstract_methods(self):

        # GIVEN
        fake_strategy = FakeStrategy()

        # WHEN
        choose_action_return = fake_strategy.choose_action(FakeActionSpace())

        # THEN
        assert isinstance(FakeStrategy, ABCMeta)
        assert choose_action_return is None
