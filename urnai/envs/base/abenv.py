from abc import abstractmethod
import inspect
import os
import sys
from typing import Tuple

from urnai.base.savable import Savable
from urnai.utils.returns import Done, Observation, Reward

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)


class Env(Savable):
    """
    Abstract Base Class for all environments currently supported.
    Environments are classes used to create a link between agents, models and
    the game. For cases where an environment for a game already exists, this class
    should still be used as a wrapper (e.g. implementing an environment for OpenAI gym).
    """

    def __init__(self, _id: str, render=False, reset_done=True):
        super().__init__()
        self.id = _id
        self.render = render
        self.reset_done = reset_done
        self.env_instance = None

    """
    Starts the environment. The implementation should assign the value of env_instance.
    """

    @abstractmethod
    def start(self) -> None: ...

    """
    Executes an action on the environment and returns an [Observation, Reward, Done] tuple.
    """

    @abstractmethod
    def step(self, action) -> Tuple[Observation, Reward, Done]: ...

    """
    Resets the environment. This method should return an Observation, since it's used by the
    Trainer to get the first Observation.
    """

    @abstractmethod
    def reset(self) -> Observation: ...

    """
    Closes the environment.
    """

    @abstractmethod
    def close(self) -> None: ...
