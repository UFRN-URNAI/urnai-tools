from abc import ABC

from action_base import ActionBase

from urnai.utils.returns import List


class ChainOfActionsBase(ABC):

	"""This class works as a sequence of actions"""
	
	action_list: List[ActionBase]
	
	def __init__(self):
		self.action_list = []
	
	def next(self):
		# Executing the next action in the sequence
		
		if self.get_len() == 0: 
			return None
	
		first_action = self.action_list[0]
		
		if first_action.is_complete():
			self.action_list.pop(0)
			self.next()
		else:
			first_action.run()
	
	def check(self) -> bool:
		# Checking all of the actions in the sequence
		
		return all(action.check() for action in self.action_list)

	def get_len(self) -> int:
		return len(self.action_list)
