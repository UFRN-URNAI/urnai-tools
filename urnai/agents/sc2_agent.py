import os
import sys

from urnai.agents.base.abagent import Agent
from urnai.agents.rewards.abreward import RewardBuilder
from urnai.models.base.abmodel import LearningModel
from urnai.utils import error

sys.path.insert(0, os.getcwd())


class SC2Agent(Agent):
    def __init__(self, model: LearningModel, reward_builder: RewardBuilder):
        super(SC2Agent, self).__init__(model, reward_builder)
        self.reward = 0
        self.episodes = 0
        self.steps = 0

    def reset(self, episode=0):
        super(SC2Agent, self).reset(episode)
        self.episodes += 1

    def step(self, obs, done, is_testing=False):
        self.steps += 1

        if self.action_wrapper.is_action_done():
            current_state = self.build_state(obs)
            excluded_actions = self.action_wrapper.get_excluded_actions(obs)
            predicted_action_idx = self.model.choose_action(current_state, excluded_actions,
                                                            is_testing)
            self.previous_action = predicted_action_idx
            self.previous_state = current_state
        selected_action = [self.action_wrapper.get_action(self.previous_action, obs)]

        try:
            ...
            # action_id = selected_action[0].function
        except Exception:
            raise error.ActionError(
                'Invalid function structure. Function name: %s.' % selected_action[0])
        return selected_action
