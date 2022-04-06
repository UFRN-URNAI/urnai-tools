import os
import sys

from urnai.agents.base.abagent import Agent
from urnai.agents.rewards.abreward import RewardBuilder
from urnai.models.base.abmodel import LearningModel

sys.path.insert(0, os.getcwd())


class GenericAgent(Agent):

    def __init__(self, model: LearningModel, reward_builder: RewardBuilder):
        super(GenericAgent, self).__init__(model, reward_builder)
        self.pickle_black_list = ['model']

    def step(self, obs, done, is_testing=False):
        if self.action_wrapper.is_action_done():
            current_state = self.build_state(
                obs)  # Builds current state (happens before executing the action on env)

            excluded_actions = self.action_wrapper.get_excluded_actions(obs)
            current_action_idx = self.model.choose_action(
                current_state, excluded_actions,
                is_testing)  # Gets an action from the model using the current state

            # Updates previous_action and previous_state to be used in self.learn()
            self.previous_action = current_action_idx
            self.previous_state = current_state

        return self.action_wrapper.get_action(self.previous_action,
                                              obs)  # Returns the decoded action from action_wrapper

    # def play(self, obs):
    #     if self.action_wrapper.is_action_done():
    #         current_state = self.build_state(obs)
    #         excluded_actions = self.action_wrapper.get_excluded_actions(obs)
    #         predicted_action_idx = self.model.predict(current_state, excluded_actions)
    #         self.previous_action = predicted_action_idx
    #         self.previous_state = current_state
    #     return self.action_wrapper.get_action(self.previous_action, obs)
