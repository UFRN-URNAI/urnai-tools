from abc import ABC, abstractmethod

from urnai.utils.returns import ActionIndex, List


class ActionBase(ABC):
		
	@abstractmethod
	def get_action(self, action_idx: ActionIndex, obs):
		"""
		Receives an action index as a parameter and returns the corresponding action from the 
		available actions. This method should return an action
		that can be used by the environment's step method.
		"""
		...
	
	@abstractmethod
	def clear_queue(self):
		...
	
	@abstractmethod
	def is_queue_empty(self) -> bool:
		...
	

