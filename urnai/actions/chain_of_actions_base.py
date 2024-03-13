from abc import ABC, abstractmethod

from action_base import ActionBase

from urnai.utils.returns import List


class ChainOfActionsBase(ABC):

	"""This class works as a sequence of actions"""
	
	action_list = []
	
	@abstractmethod
	def merge(self, chain_of_actions):
		"""Merging a list of sequences of actions into self"""
		...
	
	def next(self):
		# Executing the next action in the sequence
		
		if self.get_len() == 0: 
			return None
	
		first_action = action_list[0]
		
		if first_action.is_complete():
			action_list.pop(0)
			self.next()
		else:
			first_action.run()
	
	def check(self) -> bool:
		# Checking all of the actions in the sequence
		
		actions = self.get_actions()
		return all(action.check() for action in actions)

	def get_len(self) -> int:
		return len(self.get_actions())
