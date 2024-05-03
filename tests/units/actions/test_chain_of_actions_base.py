import unittest
from abc import ABCMeta

from urnai.actions.chain_of_actions_base import ChainOfActionsBase


class TestChainOfActionsBase(unittest.TestCase):

	def test_abstract_methods(self):
		ChainOfActionsBase.__abstractmethods__ = set()

		class FakeChainOfActions(ChainOfActionsBase):
			def __init__(self):
				super().__init__()

		f = FakeChainOfActions()
		get_action_return = f.get_action(0)
		check_return = f.check()
		len_return = f.len
		assert isinstance(ChainOfActionsBase, ABCMeta)
		assert get_action_return is None
		assert check_return is True
		assert (len_return == 0) is True
