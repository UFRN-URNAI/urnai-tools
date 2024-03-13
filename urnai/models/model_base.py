from abc import ABC, abstractmethod

from urnai.utils.returns import ActionIndex


class ModelBase(ABC):

	learning_data = {}
	
	@abstractmethod
	def learn(self, s, a, r, s_):
		"""Learning strategy"""
		...
	
	@abstractmethod
	def predict(self, state) -> ActionIndex:
		"""Returns the best action for this give state"""
		...
	
	@abstractmethod
	def save(self):
		"""Saves the learning data in a file"""
		...
		
	@abstractmethod
	def load(self):
		"""Loads learning data"""
		...
