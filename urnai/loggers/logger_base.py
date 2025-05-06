from abc import ABC, abstractmethod


class LoggerBase(ABC):
    """
    Base class for all loggers.
    """

    @abstractmethod
    def set_mode(self, train: bool) -> None:
        """
        Set the mode of the logger.
        :param train: True for training mode, False for evaluation mode.
        """
        ...
    
    @abstractmethod
    def log(self, **kwargs) -> None:
        """
        Log a message.
        """
        raise NotImplementedError("Subclasses must implement this method.")