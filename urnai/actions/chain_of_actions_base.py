from abc import ABC

from action_base import ActionBase

from urnai.utils.returns import List


class ChainOfActionsBase(ABC):

	"""This class works as a sequence of actions"""
	
	action_list: List[ActionBase]
	
	def __init__(self):
		self.action_list = []
	
	def get_action(self, action_index):
		return self.action_list[action_index]
	
	def check(self) -> bool:
		""" Checking all of the actions in the sequence """
		
		return all(action.check() for action in self.action_list)

	@property
	def len(self) -> int:
		return len(self.action_list)
