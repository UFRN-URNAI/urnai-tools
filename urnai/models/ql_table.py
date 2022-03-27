from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import StateBuilder
import numpy as np
import pandas as pd

from .base.abmodel import LearningModel


class QLearning(LearningModel):

    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder,
                 learning_rate=0.1,
                 gamma=0.95, epsilon=1, epsilon_min=0.01, epsilon_decay=0.996, name='Q-learning'):
        super(QLearning, self).__init__(action_wrapper, state_builder, gamma, learning_rate, name)
        self.__set_hyperparameters(learning_rate, gamma, epsilon, epsilon_min, epsilon_decay)
        self._q_table = pd.DataFrame(columns=range(self.action_size), dtype=np.float64)

    def __set_hyperparameters(self, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay):
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay

    def learn(self, current_state, current_action, reward, next_state, done, is_last_step: bool):

        current_state_str = str(current_state)
        next_state_str = str(next_state)

        self.__check_state_exists(current_state_str)
        self.__check_state_exists(next_state_str)

        old_value = self._q_table.loc[current_state_str, current_action]

        max_action = np.max(self._q_table.loc[next_state_str])

        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (
                    reward + self.gamma * max_action)
        self._q_table.loc[current_state_str, current_action] = new_value

        if done or is_last_step:
            self.__update_epsilon()

    def __check_state_exists(self, state_str):
        if state_str not in self._q_table.index:
            self._q_table.loc[state_str] = [0] * len(self.actions)

    def __update_epsilon(self):
        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay

    def choose_action(self, state, excluded_actions=[], is_testing=False):
        if is_testing:
            return self.predict(state, excluded_actions)
        else:
            if np.random.uniform(0, 1) < self._epsilon:  # 0.01
                return np.random.choice(self.action_size)
            else:
                return self.predict(state, excluded_actions)

    def predict(self, state, excluded_actions=[]):
        state_str = str(state)

        self.__check_state_exists(state_str)

        return self._q_table.loc[state_str, :].idxmax()
