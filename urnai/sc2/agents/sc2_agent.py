import os
import random
import sys

import numpy as np

from urnai.actions.action_space_base import ActionSpaceBase
from urnai.agents.agent_base import AgentBase
from urnai.models.model_base import ModelBase
from urnai.rewards.reward_base import RewardBase
from urnai.states.state_base import StateBase

sys.path.insert(0, os.getcwd())


class SC2Agent(AgentBase):
    def __init__(self, action_space : ActionSpaceBase, state_builder : StateBase, 
                 model: ModelBase, reward_builder: RewardBase, epsilon_start=1.0, 
                 epsilon_min=0.01, epsilon_decay_rate=0.995, 
                 per_episode_epsilon_decay=False, epsilon_linear_decay=False,
                 epsilon_decay_ep_start=0):
        super().__init__(action_space, state_builder, model, reward_builder)
        # self.reward = 0
        self.episodes = 0
        self.steps = 0

        # EXPLORATION PARAMETERS FOR EPSILON GREEDY STRATEGY
        self.epsilon_greedy = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = epsilon_decay_rate
        self.per_episode_epsilon_decay = per_episode_epsilon_decay
        self.epsilon_linear_decay = epsilon_linear_decay
        self.epsilon_decay_ep_start = epsilon_decay_ep_start


    def reset(self, episode=0):
        super().reset(episode)
        self.episodes += 1
        if self.per_episode_epsilon_decay and episode >= self.epsilon_decay_ep_start:
            self.decay_epsilon()

    def step(self, obs, done, is_training=True):
        self.steps += 1

        if self.action_space.is_action_done():
            current_state = self.state_space.update(obs)
            excluded_actions = self.action_space.get_excluded_actions(obs)
            chosen_action_idx = self.choose_action(current_state, excluded_actions,
                                                    is_training)
            self.previous_action = chosen_action_idx
            self.previous_state = current_state
        selected_action = [self.action_space.get_action(self.previous_action, obs)]

        # try:
        #     ...
        #     # action_id = selected_action[0].function
        # except Exception:
        #     raise error.ActionError(
        #         'Invalid function structure. Function name: %s.' % selected_action[0])
        return selected_action

    def choose_action(self, state, excluded_actions, is_training=True):
        if is_training:
            if np.random.rand() <= self.epsilon_greedy:
                random_action = random.choice(self.action_space.get_actions())
                # Removing excluded actions
                while random_action in excluded_actions:
                    random_action = random.choice(self.action_space.get_actions())
                return random_action
            else:
                return self.model.predict(state, excluded_actions)
        else:
            return self.model.predict(state, excluded_actions)
        
    def learn(self, obs, reward, done) -> None:
        """
        If it is not the very first step in an episode, this method will
        call the model's learn method.
        """
        if self.previous_state is not None:
            next_state = self.state_space.update(obs)
            self.model.learn(self.previous_state, self.previous_action,
                              reward, next_state, done)
            if not self.per_episode_epsilon_decay:
                self.decay_epsilon()

    def decay_epsilon(self):
        """
        Implements the epsilon greedy strategy, effectivelly lowering the current
        epsilon greedy value by multiplying it by the epsilon_decay_rate
        (the higher the value, the less it lowers the epsilon_decay).
        """
        if self.epsilon_linear_decay:
            if self.epsilon_greedy > self.epsilon_min:
                self.epsilon_greedy -= (1 - self.epsilon_decay_rate)
        else:
            if self.epsilon_greedy > self.epsilon_min:
                self.epsilon_greedy *= self.epsilon_decay_rate