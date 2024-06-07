from abc import ABC, abstractmethod


class ActionSpaceBase(ABC):
    """
    ActionSpace works as an extra abstraction layer used by the agent
    to select actions. This means the agent doesn't select actions from
    action_set, but from ActionSpace.
    
    This class is responsible of telling the agent which actions it can 
    use and which ones are excluded from selection. It can also force the
    agent to use certain actions by combining them into multiple steps.
    """

    @abstractmethod
    def is_action_done(self) -> bool:
        """
        Some agents must do multiple steps for a single action before they
        can choose another one. This method should implement the logic to
        tell whether the current action is done or not. That is, if all the 
        steps for an action are complete.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Contains logic for resetting the action_space. This is used mostly
        for agents that require multiple steps for a single action.
        """
        ...

    @abstractmethod
    def get_actions(self) -> list[int]:
        """Returns all the actions that the agent can choose from."""
        ...

    @abstractmethod
    def get_excluded_actions(self, obs) -> list[int]:
        """Returns a subset of actions that can't be chosen by the agent."""
        ...

    @abstractmethod
    def get_action(self, action_idx: int, obs):
        """
        Receives an action index as a parameter and returns the corresponding
        action from the available actions. This method should return an action
        that can be used by the environment's step method.
        """
        pass

    def get_named_actions(self) -> list[str]:
        """Returns the names of all the actions that the agent can choose from."""
        names = []
        for action in self.get_actions():
            names.append(action.name)
        return names

    @property
    def size(self) -> int:
        return len(self.get_actions())
