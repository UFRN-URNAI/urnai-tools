from abc import abstractmethod
import os
import sys

from urnai.agents.rewards.abreward import RewardBuilder
from urnai.base.savable import Savable
from urnai.models.base.abmodel import LearningModel

sys.path.insert(0, os.getcwd())


class Agent(Savable):

    def __init__(self, model: LearningModel, reward_builder: RewardBuilder):
        super().__init__()
        self.model = model
        self.action_wrapper = model.action_wrapper
        self.state_builder = model.state_builder
        self.previous_action = None
        self.previous_state = None
        self.reward_builder = reward_builder

        self.pickle_black_list = ['model', 'action_wrapper', 'state_builder', 'reward_builder']

    def build_state(self, obs):
        """
        Calls the build_state method from the state_builder, effectivelly returning the state of
        the game environment through the  lens of the state_builder.
        """
        return self.state_builder.build_state(obs)

    def get_reward(self, obs, reward, done):
        """
        Calls the get_reward method from the reward_builder, effectivelly returning the reward
        value.
        """
        return self.reward_builder.get_reward(obs, reward, done)

    def get_state_dim(self):
        """Returns the dimensions of the state builder"""
        return self.state_builder.get_state_dim()

    def reset(self, episode=0):
        """
        Resets some Agent class variables, such as previous_action and previous_state.
        Also, calls the respective reset methods for the action_wrapper and model.
        """
        self.previous_action = None
        self.previous_state = None
        self.action_wrapper.reset()
        self.model.ep_reset(episode)
        self.reward_builder.reset()
        self.state_builder.reset()

    def learn(self, obs, reward, done):
        """
        If it is not the very first step in an episode, this method will call the model's learn
        method.
        """
        if self.previous_state is not None:
            next_state = self.build_state(obs)
            self.model.learn(self.previous_state, self.previous_action, reward, next_state, done)

    @abstractmethod
    def step(self, obs, done, is_testing=False):
        """
        This method should:
        1) Build a State using obs
        2) Use the state that was built to get an ActionIndex from the Agent's model
        3) Update self.previous_state with the current state and self.previous_action with the
          ActionIndex
        4) Return an Action from the Agent's ActionWrapper by using the ActionIndex from step 2
        """
        pass

    def save_extra(self, save_path):
        """
        Implements the save_extra method from the Savable class.
        In the Agent class, this method will call the model's save method.
        """
        self.model.save(save_path)

    def load_extra(self, load_path):
        """
        Implements the load_extra method from the Savable class.
        In the Agent class, this method will call the model's load method.
        """
        self.model.load(load_path)
