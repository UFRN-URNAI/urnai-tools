import tensorflow as tf
import numpy as np
import random
import os
from datetime import datetime
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from models.ddqn_keras import DDQNKeras
from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import StateBuilder
from .model_builder import ModelBuilder
from urnai.utils.error import IncoherentBuildModelError
from urnai.utils.error import UnsupportedBuildModelLayerTypeError

class DDQNKerasMO(DDQNKeras):
    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, gamma=0.99,
                    learning_rate=0.001, learning_rate_min=0.0001, learning_rate_decay=0.99995, learning_rate_decay_ep_cutoff = 0,
                    name='DDQN', epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.99995, per_episode_epsilon_decay=False,
                    batch_size=32, use_memory=True, memory_maxlen=50000, min_memory_size=1000, build_model = ModelBuilder.DEFAULT_BUILD_MODEL, update_target_every=5, 
                    seed_value=None, cpu_only=False):
        super(DDQNKerasMO, self).__init__(action_wrapper, state_builder, gamma=gamma, use_memory=use_memory,  name=name,
                                        learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, 
                                        learning_rate_min=learning_rate_min, learning_rate_decay_ep_cutoff= learning_rate_decay_ep_cutoff,
                                        epsilon_start=epsilon_start, epsilon_min=epsilon_min, 
                                        epsilon_decay=epsilon_decay, per_episode_epsilon_decay=per_episode_epsilon_decay,
                                        seed_value=seed_value, cpu_only=cpu_only)

        self.build_model = build_model
        self.loss = 0

        # Main model, trained every step
        self.model = self.make_model()
        # Target model, used in .predict every step (does not get update every step)
        self.target_model = self.make_model()
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0
        self.update_target_every = update_target_every

        if self.use_memory:
            self.memory = deque(maxlen=memory_maxlen)
            self.memory_maxlen = memory_maxlen
            self.min_memory_size = min_memory_size
            self.batch_size = batch_size

    def memory_learn(self, s, a, r, s_, done):
        self.memorize(s, a, r, s_, done)
        if len(self.memory) < self.min_memory_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        # array of initial states from the minibatch
        current_states = np.array([transition[0] for transition in minibatch])
        # removing undesirable dimension created by np.array
        current_states = np.squeeze(current_states)
        # array of Q-Values for our initial states
        current_qs_list = self.model.predict(current_states)

        # array of states after step from the minibatch
        next_current_states = np.array([transition[3] for transition in minibatch])
        next_current_states = np.squeeze(next_current_states)
        # array of Q-values for our next states
        next_qs_list = self.target_model.predict(next_current_states)

        # inputs is going to be filled with all current states from the minibatch
        # targets is going to be filled with all of our outputs (Q-Values for each action)
        inputs = []
        targets = []

        for index, (state, actions, reward, next_state, done) in enumerate(minibatch):
            for j, (action) in enumerate(actions):
                # if this step is not the last, we calculate the new Q-Value based on the next_state
                if not done:
                    max_next_q = np.max(next_qs_list[index][self.action_wrapper.multi_output_ranges[j]:self.action_wrapper.multi_output_ranges[j+1]])
                    # new Q-value is equal to the reward at that step + discount factor * the max q-value for the next_state
                    new_q = reward + self.gamma * max_next_q
                else:
                    # if this is the last step, there is no future max q value, so the new_q is just the reward
                    new_q = reward

                current_qs = current_qs_list[index]
                current_qs[action] = new_q

            inputs.append(state)
            targets.append(current_qs)

        np_inputs = np.squeeze(np.array(inputs))
        np_targets = np.array(targets)

        self.loss = self.model.fit(np_inputs, np_targets, batch_size=self.batch_size, verbose=0, shuffle=False, callbacks=self.tensorboard_callback)

        # If it's the end of an episode, increase the target update counter
        if done:
            self.target_update_counter += 1

        # If our target update counter is greater than update_target_every we will update the weights in our target model
        if self.target_update_counter > self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        # if our epsilon rate decay is set to be done every step, we simply decay it. Otherwise, this will only be done
        # at the end of every episode, on self.ep_reset() which is in our LearningModel base class
        if not self.per_episode_epsilon_decay:
            self.decay_epsilon()


    def no_memory_learn(self, s, a, r, s_, done):

        # Q-Value for our initial states
        current_qs = self.model.predict(s)

        # Q-value for our next states
        next_qs_list = self.target_model.predict(s_)

        # if this step is not the last, we calculate the new Q-Value based on the next_state
        if not done:
            max_next_q = np.max(next_qs_list[0])
            # new Q-value is equal to the reward at that step + discount factor * the max q-value for the next_state
            new_q = r + self.gamma * max_next_q
        else:
            # if this is the last step, there is no future max q value, so we the new_q is just the reward
            new_q = r

        current_qs[0][a] = new_q

        inputs = s
        targets = current_qs

        self.loss = self.model.fit(inputs, targets, verbose=0, callbacks=self.tensorboard_callback)

        # If it's the end of an episode, increase the target update counter
        if done:
            self.target_update_counter += 1

        # If our target update counter is greater than update_target_every we will update the weights in our target model
        if self.target_update_counter > self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        # if our epsilon rate decay is set to be done every step, we simply decay it. Otherwise, this will only be done
        # at the end of every episode, on self.ep_reset() which is in our LearningModel base class
        if not self.per_episode_epsilon_decay:
            self.decay_epsilon()


    def choose_action(self, state, excluded_actions=[], is_testing=False):
        if is_testing:
            return self.predict(state, excluded_actions)
        
        else:
            if np.random.rand() <= self.epsilon_greedy:
                random_action = []

                for i in range(len(self.action_wrapper.multi_output_ranges)-1):
                    random_action.append(random.choice(self.actions[self.action_wrapper.multi_output_ranges[i]:self.action_wrapper.multi_output_ranges[i+1]]))
                
                return random_action
            else:
                return self.predict(state, excluded_actions)

    def predict(self, state, excluded_actions=[]):
        '''
        model.predict returns an array of arrays, containing the Q-Values for the actions. This function should return the
        corresponding action with the highest Q-Value.
        '''
        q_values = self.model.predict(state)[0]
        culmulative_range = 0
        action_idx = []
        for i in range(len(self.action_wrapper.multi_output_ranges)-1):
            culmulative_range = self.action_wrapper.multi_output_ranges[i]
            action_idx.append(culmulative_range + int(np.argmax(
                            q_values[self.action_wrapper.multi_output_ranges[i]:self.action_wrapper.multi_output_ranges[i+1]] )))

        return action_idx