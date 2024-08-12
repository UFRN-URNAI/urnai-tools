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

	@abstractmethod
	def choose_action(self, state, excluded_actions, is_testing=False) -> int:
		"""
		Implements the logic for choosing an action while training and while 
		testing an Agent. For most Reinforcement Learning Algorithms, this method 
		will choose an action directly When is_testing=True, and will implement the 
		exploration algorithm for when is_testing=False.
		One such exploration algorithm commonly used it the epsilon greedy strategy.
		"""
		pass

	def save(self, persist_path) -> None:
		self.persistence.save(persist_path)

	def load(self, persist_path) -> None:
		self.persistence.load(persist_path)