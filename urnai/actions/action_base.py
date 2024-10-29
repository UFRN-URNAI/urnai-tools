from abc import ABC, abstractmethod
from typing import Any


class ActionBase(ABC):
	__id__ = None
	
	@abstractmethod
	def run(*args) -> Any:
		"""Executing the action"""
		...
