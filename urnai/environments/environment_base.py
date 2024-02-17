from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class EnvironmentBase(ABC):
    """
    Abstract Base Class for all environments currently supported.

    Environments are classes used to create a link between agents, models and
    the game. For cases where an environment for a game already exists, this class
    should still be used as a wrapper (e.g. implementing an environment for OpenAI gym).
    """

    def __init__(self, id: str, render=False, reset_done=True):
        self.id = id
        self.render = render
        self.reset_done = reset_done
        self.env_instance = None

    @abstractmethod
    def start(self) -> None:
        """
        Start the environment.
        The implementation should assign the value of env_instance.
        """
        ...

    @abstractmethod
    def step(self, action) -> Tuple[List[Any], int, bool]:
        """
        Execute an action on the environment and returns an
        [Observation, Reward, Done] tuple.
        """
        ...

    @abstractmethod
    def reset(self) -> List[Any]:
        """
        Reset the environment.
        This method should return an Observation, since it's used by the
        Trainer to get the first Observation.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """
        Close the environment.
        """
        ...
