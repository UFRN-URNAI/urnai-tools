from abc import ABC, abstractmethod


class RewardBase(ABC):
    """
    Every Agent needs to own an instance of this base class in order to calculate
    its rewards. So every time we want to create a new agent,
    we should either use an existing RewardBase implementation or create a new one.
    """

    @abstractmethod
    def get(
        self,
        obs: list[list],
        default_reward: int,
        terminated: bool, 
        truncated: bool
    ) -> int: 
        raise NotImplementedError("Get method not implemented. You should implement " +
                                  "it in your RewardBase subclass.")

    @abstractmethod
    def reset(self) -> None: ...
