import os
import sys

from urnai.actions.action_space_base import ActionSpaceBase
from urnai.agents.agent_base import AgentBase
from urnai.models.model_base import ModelBase
from urnai.rewards.reward_base import RewardBase
from urnai.states.state_base import StateBase

sys.path.insert(0, os.getcwd())


class SC2Agent(AgentBase):
    def __init__(self, action_space : ActionSpaceBase, state_builder : StateBase, 
                 model: ModelBase, reward_builder: RewardBase):
        super().__init__(action_space, state_builder, model, reward_builder)
        # self.reward = 0
        self.episodes = 0
        self.steps = 0

    def reset(self, episode=0):
        super().reset(episode)
        self.episodes += 1

    def step(self, obs, done, is_training=True):
        self.steps += 1

        if self.action_space.is_action_done():
            current_state = self.state_space.update(obs)
            excluded_actions = self.action_space.get_excluded_actions(obs)
            predicted_action_idx = self.model.choose_action(current_state, 
                                                            excluded_actions,
                                                            is_training)
            self.previous_action = predicted_action_idx
            self.previous_state = current_state
        selected_action = [self.action_space.get_action(self.previous_action, obs)]

        # try:
        #     ...
        #     # action_id = selected_action[0].function
        # except Exception:
        #     raise error.ActionError(
        #         'Invalid function structure. Function name: %s.' % selected_action[0])
        return selected_action
