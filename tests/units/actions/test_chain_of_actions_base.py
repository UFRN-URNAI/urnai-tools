import unittest
from abc import ABCMeta

from urnai.actions.chain_of_actions_base import ChainOfActionsBase


class FakeChainOfActions(ChainOfActionsBase):
	ChainOfActionsBase.__abstractmethods__ = set()
	
	def __init__(self):
		super().__init__()

class TestChainOfActionsBase(unittest.TestCase):

	def test_abstract_methods(self):

		# GIVEN
		fake_chain_of_actions = FakeChainOfActions()

		# WHEN
		get_action_return = fake_chain_of_actions.get_action(0)
		length_return = fake_chain_of_actions.length

		# THEN
		assert isinstance(ChainOfActionsBase, ABCMeta)
		assert get_action_return is None
		assert (length_return == 0) is True
