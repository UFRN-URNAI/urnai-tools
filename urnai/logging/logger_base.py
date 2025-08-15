from abc import ABC, abstractmethod
from typing import get_type_hints

from urnai.logging.logging_mode_base import LoggingModeBase


class LoggerBase(ABC):
    """
    Base class for all loggers.
    """
    
    _mode_cls: LoggingModeBase = LoggingModeBase
    """Class variable that defines the type of logging mode to use.
    This should be overridden by subclasses to specify their own LoggingModeBase enum class.
    Defaults to base LoggingModeBase."""

    def __init__(self, mode: str | LoggingModeBase):
        self.set_mode(mode)

    def set_mode(self, mode: str | LoggingModeBase) -> None:
        """
        Set the mode of the logger.
        :param mode: A LoggingMode instance representing either TRAINING or EVALUATION mode.
        """
        self._mode_cls.check(mode)
        self.mode = mode if isinstance(mode, self._mode_cls) else self._mode_cls[mode.upper()]
    
    @abstractmethod
    def log(self, **kwargs) -> None:
        """
        Log a message.
        """
        raise NotImplementedError("The 'log' method must be implemented.")
