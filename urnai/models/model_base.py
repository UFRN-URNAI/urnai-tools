from abc import ABC, abstractmethod
from typing import Dict


class ModelBase(ABC):

	learning_data: Dict
	
	def __init__(self):
		self.learning_data = {}
		self.persistence = None
	
	@abstractmethod
	def learn(self, current_state, action, reward, next_state):
		"""Learning strategy"""
		...
	
	@abstractmethod
	def predict(self, state) -> int:
		"""Returns the best action for this given state"""
		...

	def save(self, persist_path) -> None:
		self.persistence.save(persist_path)

	def load(self, persist_path) -> None:
		self.persistence.load(persist_path)