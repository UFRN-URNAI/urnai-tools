from abc import ABC, abstractmethod

from urnai.actions.action_space_base import ActionSpaceBase
from urnai.models.model_base import ModelBase
from urnai.rewards.reward_base import RewardBase
from urnai.states.state_base import StateBase


class AgentBase(ABC):

    def __init__(self, action_space : ActionSpaceBase,
                 state_space : StateBase,
                 model : ModelBase,
                 reward : RewardBase):

        self.action_space = action_space
        self.state_space = state_space
        self.model = model
        self.reward = reward

        self.previous_action = None
        self.previous_state = None

        self.attr_block_list = ['model', 'action_space',
                                'state_space', 'reward']

    @abstractmethod
    def step(self) -> None:
        ...

    # @abstractmethod
    # def choose_action(self, action_space : ActionSpaceBase) -> ActionBase:
    #     """
    #     Method that contains the agent's strategy for choosing actions
    #     """
    #     ...

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
        self.reward.reset()
        self.state_space.reset()

    def learn(self, obs, reward, done) -> None:
        """
        If it is not the very first step in an episode, this method will
        call the model's learn method.
        """
        if self.previous_state is not None:
            next_state = self.state_space.update(obs)
            self.model.learn(self.previous_state, self.previous_action,
                              reward, next_state, done)
            
    def save(self, savepath) -> None:
        if (self.model.persistence == None):
            raise 'No persistence set in model'
        
        self.model.persistence.save(savepath)
