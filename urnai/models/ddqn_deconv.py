from collections import deque
import enum
import os
import random
from re import I
from contextlib import redirect_stdout

from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import StateBuilder
from urnai.base.savable import Savable
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import utils

class DDQN_Deconv(Savable):
    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, gamma=0.99,
                 learning_rate=0.001, lr_min=0.0001, lr_decay=0.99995, lr_decay_min_ep=0, lr_linear_decay=False,
                 epsilon_start=1.0, epsilon_min=0.001, epsilon_decay=0.9995, per_episode_epsilon_decay=True, epsilon_linear_decay=False,
                 batch_size=32, memory_maxlen=50000, min_memory_size=128, update_target_every=5, 
                 model_layers = [30, 30]):

        self.pickle_black_list = ["model", "target_model", "target_update_counter", "update_target_every"]

        self.loss = 0
        self.mse = 0
        self.gamma = gamma
        self.learning_rate=learning_rate
        self.lr_min = lr_min
        self.lr_decay = lr_decay
        self.lr_decay_min_ep = lr_decay_min_ep
        self.lr_linear_decay = lr_linear_decay

        self.model_layers = model_layers
        self.action_wrapper = action_wrapper
        self.state_builder = state_builder
        self.actions = action_wrapper.get_actions()
        self.action_size = action_wrapper.get_action_space_dim()
        self.state_size = state_builder.get_state_dim()

        self.epsilon_greedy = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = epsilon_decay
        self.per_episode_epsilon_decay = per_episode_epsilon_decay
        self.epsilon_linear_decay = epsilon_linear_decay

        # Main model, trained every step
        self.model = self.make_model()
        # Target model, used in .predict every step (does not get update every step)
        self.target_model = self.make_model()
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0
        self.update_target_every = update_target_every

        self.memory = deque(maxlen=memory_maxlen)
        self.memory_maxlen = memory_maxlen
        self.min_memory_size = min_memory_size
        self.batch_size = batch_size

    def make_model(self):
        model = models.Sequential()
        model.add(layers.Input((self.state_size,)))
        # for layer_size in self.model_layers:
        #     model.add(layers.Dense(layer_size, activation=activations.relu))
        
        model.add(layers.Reshape((10, 10, 1)))
        model.add(layers.Conv2D(16, 3, activation=activations.relu, padding='same'))
        model.add(layers.Conv2D(16, 3, activation=activations.relu, padding='same'))
        model.add(layers.Conv2D(1, 3, activation=activations.relu, padding='same'))

        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mse'])
        model.summary()
        return model

    def learn(self, s, a, r, s_, done):
        self.memorize(s, a, r, s_, done)
        if len(self.memory) < self.min_memory_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        # array of initial states from the minibatch
        current_states = np.array([transition[0] for transition in minibatch])
        # removing undesirable dimension created by np.array
        current_states = np.squeeze(current_states)
        # array of Q-Values for our initial states
        current_qs_list = self.model(current_states).numpy()

        # array of states after step from the minibatch
        next_current_states = np.array([transition[3] for transition in minibatch])
        next_current_states = np.squeeze(next_current_states)
        # array of Q-values for our next states
        next_qs_list = tf.Variable(self.model(next_current_states)).numpy()
        targ_qs_list = tf.Variable(self.target_model(next_current_states)).numpy()

        # inputs is going to be filled with all current states from the minibatch
        # targets is going to be filled with all of our outputs (Q-Values for each action)
        inputs = []
        targets = []

        for index, (state, action, reward, next_state, done) in enumerate(minibatch):
            # if this step is not the last, we calculate the new Q-Value based on the next_state
            if not done:
                max_next_q = np.unravel_index(np.argmax(next_qs_list[index]), next_qs_list[index].shape)
                # new Q-value is equal to the reward at that step + discount factor * the
                # max q-value for the next_state
                new_q = reward + self.gamma * targ_qs_list[index][max_next_q]
            else:
                # if this is the last step, there is no future max q value, so the new_q is
                # just the reward
                new_q = reward

            current_qs_list[index][action] = new_q

            # current_qs = current_qs_list[index]
            # current_qs[action] = new_q

            # inputs.append(state)
            # targets.append(current_qs_list)

        np_inputs = current_states
        np_targets = current_qs_list

        history = self.model.fit(np_inputs, np_targets, batch_size=self.batch_size, verbose=0)

        self.loss = history.history['loss'][0]
        self.mse = history.history['mse'][0]

        # If it's the end of an episode, increase the target update counter
        if done:
            self.target_update_counter += 1

        # If our target update counter is greater than update_target_every we will
        # update the weights in our target model
        if self.target_update_counter > self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        # if our epsilon rate decay is set to be done every step, we simply decay it.
        # Otherwise, this will only be done
        # at the end of every episode, on self.ep_reset() which is in our LearningModel base class
        if not self.per_episode_epsilon_decay:
            self.decay_epsilon()

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, excluded_actions=[], is_testing=False):
        # Verifies if we are running a test (Evaluating our agent)
        if is_testing:
            return self.predict(state, excluded_actions)
        # If we are not testing (therefore we are training), evaluate epsilon greedy strategy
        else:
            if np.random.rand() <= self.epsilon_greedy:
                random_action = np.random.choice(self.actions, self.action_wrapper.get_action_shape(), replace=False)
                pos = np.unravel_index(np.argmax(random_action, axis=None), random_action.shape)
                return pos
            else:
                return self.predict(state, excluded_actions)

    def predict(self, state, excluded_actions=[]):
        """
        model.predict returns an array of arrays, containing the Q-Values for the actions.
        This function should return the corresponding action with the highest Q-Value.
        """
        q_values = np.squeeze(self.model(state).numpy()[0])
        pos = np.unravel_index(np.argmax(q_values, axis=None), q_values.shape)

        return pos

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

    def decay_lr(self):
        """
        Implements a strategy for gradually lowering the learning rate of our model.
        This method works very similarly to decay_epsilon(), lowering the learning rate
        by multiplying it by the learnin_rate_decay.
        """
        if self.lr_linear_decay:
            if self.learning_rate > self.lr_min:
                self.learning_rate -= (1 - self.lr_decay)
        else:
            if self.learning_rate > self.lr_min:
                self.learning_rate *= self.lr_decay

    def ep_reset(self, episode=0):
        """
        This method is mainly used to enact the decay_epsilon and decay_lr
        at the end of every episode.
        """
        if self.per_episode_epsilon_decay:
            self.decay_epsilon()

        if episode > self.lr_decay_min_ep and self.lr_decay != 1:
            self.decay_lr()

    def save_extra(self, persist_path):
        self.model.save(self.get_full_persistance_path(persist_path))
        with open(self.get_full_persistance_path(persist_path) + 'model_summary.txt', 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

    def load_extra(self, persist_path):
        exists = os.path.exists(self.get_full_persistance_path(persist_path))

        if (exists):
            self.model = models.load_model(self.get_full_persistance_path(persist_path))
            self.target_model = models.load_model(self.get_full_persistance_path(persist_path))