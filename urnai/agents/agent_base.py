from abc import ABC, abstractmethod

from urnai.actions.action_space_base import ActionSpaceBase
from urnai.models.model_base import ModelBase
from urnai.states.state_base import StateBase


class AgentBase(ABC):

    def __init__(self, action_space : ActionSpaceBase,
                 state_space : StateBase,
                 model : ModelBase,
                 reward_builder):

        self.action_space = action_space
        self.state_space = state_space
        self.model = model
        self.reward_builder = reward_builder

        self.previous_action = None
        self.previous_state = None

        self.attr_block_list = ['model', 'action_space',
                                'state_space', 'reward_builder']

    @abstractmethod
    def step(self) -> None:
        ...

    def reset(self, episode=0) -> None:
        """
        Resets some Agent class variables, such as previous_action
        and previous_state. Also, calls the respective reset methods
        for the action_wrapper and model.
        """
        self.previous_action = None
        self.previous_state = None
        self.action_space.reset()
        self.model.ep_reset(episode)
        self.reward_builder.reset()
        self.state_space.reset()

    def learn(self, obs, reward, done) -> None:
        """
        If it is not the very first step in an episode, this method will
        call the model's learn method.
        """
        if self.previous_state is not None:
            next_state = self.update_state(obs)
            self.model.learn(self.previous_state, self.previous_action,
                              reward, next_state, done)


    def update_state(self, obs) -> list:
        """
        Returns the state of the game environment
        """
        return self.state_space.update(obs)

    def get_reward(self, obs, reward, done) -> None:
        """
        Calls the get_reward method from the reward_builder, effectivelly
        returning the reward value.
        """
        return self.reward_builder.get_reward(obs, reward, done)

    @property
    def state_dim(self) -> int:
        """Returns the dimensions of the state builder"""
        return self.state_space.dimension
