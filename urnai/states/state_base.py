from abc import ABC, abstractmethod
from typing import Any, List


class StateBase(ABC):
    """
    Every Agent needs to own an instance of this base class 
    in order to define its State.
    So every time we want to create a new agent,
    we should either use an existing State implementation or create a new one.
    """

    @abstractmethod
    def update(self, obs) -> List[Any]:
        """
        This method receives as a parameter an Observation and returns a State, which 
        is usually a list of features extracted from the Observation. The Agent uses
        this State during training to receive a new action from its model and also to
        make it learn, that's why this method should always return a list.
        """
        pass

    @property
    @abstractmethod
    def get_state(self):
        """Returns the State currently saved."""
        pass

    @property
    @abstractmethod
    def get_dimension(self):
        """Returns the dimensions of the States returned by the update method."""
        pass

    @abstractmethod
    def reset(self):
        """Resets the State currently saved."""
        pass