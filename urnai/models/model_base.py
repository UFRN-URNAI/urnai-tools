from abc import ABC, abstractmethod
from typing import Dict

from urnai.utils.returns import ActionIndex


class ModelBase(ABC):

	learning_data: Dict
	
	def __init__(self):
		self.learning_data = {}
	
	@abstractmethod
	def learn(self, current_state, action, reward, next_state):
		"""Learning strategy"""
		...
	
	@abstractmethod
	def predict(self, state) -> ActionIndex:
		"""Returns the best action for this given state"""
		...
