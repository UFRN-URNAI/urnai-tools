from abc import ABC, abstractmethod

from urnai.utils.returns import ActionIndex, List


class ChainOfActionsBase(ABC): 

	@abstractmethod
	def get_actions(self) -> List[ActionIndex]:
		"""Returns all the actions that the agent can choose from."""
		...
	
	@abstractmethod
	def get_excluded_actions(self, obs) -> List[ActionIndex]:
		"""Returns a subset of actions that can't be chosen by the agent."""
		...
	
	def get_named_actions(self) -> List[str]:
		"""Returns the names of all the actions that the agent can choose from."""
		return None

	def get_action_space_dim(self):
		return len(self.get_actions())
