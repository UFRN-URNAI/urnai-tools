from abc import ABC, abstractmethod
from typing import Dict


class ModelBase(ABC):

	learning_data: Dict
	
	def __init__(self):
		self.learning_data = {}
	
	@abstractmethod
	def learn(self, current_state, action, reward, next_state):
		"""Learning strategy"""
		...
	
	@abstractmethod
	def predict(self, state) -> int:
		"""Returns the best action for this given state"""
		...
