import unittest
from abc import ABCMeta

from urnai.actions.action_base import ActionBase
from urnai.actions.chain_of_actions_base import ChainOfActionsBase


class FakeActionBase(ActionBase):
	ActionBase.__abstractmethods__ = set()
	
	def __init__(self):
		super().__init__()

class FakeChainOfActions(ChainOfActionsBase):
	ChainOfActionsBase.__abstractmethods__ = set()
	
	def __init__(self):
		super().__init__()

class TestChainOfActionsBase(unittest.TestCase):

	def test_abstract_methods(self):

		# GIVEN
		fake_chain_of_actions = FakeChainOfActions()

		# WHEN
		get_action_return_no_actions = fake_chain_of_actions.get_action(0)

		fake_action = FakeActionBase()
		fake_chain_of_actions.action_list.append(fake_action)
		get_action_return_has_actions = fake_chain_of_actions.get_action(0)

		length_return = fake_chain_of_actions.length

		# THEN
		assert isinstance(ChainOfActionsBase, ABCMeta)
		assert get_action_return_no_actions is None
		self.assertEqual(get_action_return_has_actions, fake_action)
		assert (length_return == 1) is True
