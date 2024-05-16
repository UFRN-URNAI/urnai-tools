from abc import ABC, abstractmethod


class EnvironmentBase(ABC):
    """
    Abstract Base Class for all environments currently supported.

    Environments are classes used to create a link between agents, models and
    the game. For cases where an environment for a game already exists, this class
    should still be used as a wrapper (e.g. implementing an environment for OpenAI gym).
    """

    def __init__(self, map_name: str, visualize=False, reset_done=True):
        self.map_name = map_name
        self.visualize = visualize
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
    def step(self, action) -> tuple[list, int, bool, bool]:
        """
        Execute an action on the environment and returns an
        [Observation, Reward, Terminated, Truncated] tuple.
        """
        ...

    @abstractmethod
    def reset(self) -> list:
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
