from abc import abstractmethod

from urnai.actions.action_base import ActionBase


class ActionBaseStrict(ActionBase):
	
	@abstractmethod
	def run(*args):
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
