from abc import ABC

from urnai.actions.action_base import ActionBase


class ChainOfActionsBase(ABC):

	"""This class works as a sequence of actions"""
	
	action_list: list[ActionBase]
	
	def __init__(self):
		self.action_list = []
	
	def get_action(self, action_index) -> ActionBase:
		if action_index < self.length:
			return self.action_list[action_index]

	@property
	def length(self) -> int:
		return len(self.action_list)
