from abc import ABC, abstractmethod


class ActionBase(ABC):
	
	def __init__(self, id):
		self.id = id
	
	@abstractmethod
	def run(self):
		"""Contains logic for executing the action"""
		...
	
	@abstractmethod
	def check(self, obs) -> bool:
		"""Returns whether the action can be executed or not"""
		...
	
	@abstractmethod
	def is_complete(self) -> bool:
		"""Returns whether the action has finished or not"""
		...
		
	

