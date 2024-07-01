from abc import ABC, abstractmethod

from urnai.actions.action_space_base import ActionSpaceBase


class StrategyBase(ABC):

    """
    This class is responsible for the decision making of the agent.
    And, for example, it might or not rely on information stored in
    instances of the class Model.
    """

    @abstractmethod
    def choose_action(action_space : ActionSpaceBase):
        ...
