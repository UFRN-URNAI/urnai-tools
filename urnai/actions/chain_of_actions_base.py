from abc import ABC

from urnai.actions.action_base import ActionBase


class ChainOfActionsBase(ABC):

	"""This class works as a sequence of actions"""
	
	action_list: list[ActionBase]
	
	def __init__(self):
		self.action_list = []
	
	def get_action(self, action_index):
		if action_index >= self.len:
			return None
		else:
			return self.action_list[action_index]
	
	def check(self) -> bool:
		""" Checking all of the actions in the sequence """
		
		return all(action.check() for action in self.action_list)

	@property
	def len(self) -> int:
		return len(self.action_list)
