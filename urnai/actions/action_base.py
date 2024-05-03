from abc import ABC, abstractmethod


class ActionBase(ABC):
	id = None
	
	@abstractmethod
	def run(self) -> None:
		"""Executing the action"""
		...
	
	@abstractmethod
	def check(self, obs) -> bool:
		"""Returns whether the action can be executed or not"""
		...
	
	@abstractmethod
	def is_complete(self) -> bool:
		"""Returns whether the action has finished or not"""
		...
