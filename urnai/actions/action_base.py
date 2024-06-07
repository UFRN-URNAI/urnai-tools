from abc import ABC, abstractmethod


class ActionBase(ABC):
	__id__ = None

	def __init__(self, name):
		self.name = name
	
	@abstractmethod
	def run(self) -> None:
		"""Executing the action"""
		...
	
	@abstractmethod
	def check(self, obs) -> bool:
		"""Returns whether the action can be executed or not"""
		...
	
	@property
	@abstractmethod
	def is_complete(self) -> bool:
		"""Returns whether the action has finished or not"""
		...
