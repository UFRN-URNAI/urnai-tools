from abc import ABC, abstractmethod

from action_base import ActionBase

from urnai.utils.returns import List


class ChainOfActionsBase(ABC):

	"""This class works as a sequence of actions"""

	@abstractmethod
	def get_actions(self) -> List[ActionBase]:
		"""Returns all the actions in the sequence."""
		...
	
	@abstractmethod
	def get_first(self) -> ActionBase:
		"""Returns the first action in the sequence"""
		...
	
	@abstractmethod
	def add_action(self, action: ActionBase):
		"""Contains logic for adding an action to the sequence"""
		...
		
	@abstractmethod
	def remove_action(self, action: ActionBase):
		"""Contains logic for removing an action from the sequence"""
		...
		
	@abstractmethod
	def clear():
		...
	
	@abstractmethod
	def merge(self, chain_of_actions):
		"""Contains logic for merging a list of sequences of actions into self"""
		...
	
	# Executing the next action in the sequence
	def next(self):
		if (self.get_len() == 0): 
			return None
	
		first_action = self.get_first()
		
		if first_action.is_complete():
			self.remove_action(first_action)
			self.next()
		else:
			first_action.run()
	
	# Checking all of the actions in the sequence
	def check(self) -> bool:
		actions = self.get_actions()
		return all(a.check() for a in actions)

	def get_len(self):
		return len(self.get_actions())
