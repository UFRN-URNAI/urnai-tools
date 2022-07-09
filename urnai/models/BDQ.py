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

class BDQ(Savable):
    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, gamma=0.99,
                 learning_rate=0.001, lr_min=0.0001, lr_decay=0.99995, lr_decay_min_ep=0, lr_linear_decay=False,
                 epsilon_start=1.0, epsilon_min=0.001, epsilon_decay=0.9995, per_episode_epsilon_decay=True, epsilon_linear_decay=False,
                 batch_size=32, memory_maxlen=50000, min_memory_size=128, update_target_every=1000, 
                 model_layers = [30, 30], use_conv=False, epsilon_decay_ep_start=0):

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
        self.act_ranges = self.action_wrapper.multi_output_ranges

        self.epsilon_greedy = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = epsilon_decay
        self.per_episode_epsilon_decay = per_episode_epsilon_decay
        self.epsilon_linear_decay = epsilon_linear_decay
        self.epsilon_decay_ep_start = epsilon_decay_ep_start
        self.use_conv = use_conv
        if use_conv:
            self.input_shape = (self.state_size[0], self.state_size[1], 1)

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
        act_f = activations.relu
        
        if self.use_conv:
            inp = layers.Input(shape=self.input_shape, name="input")
            x = layers.Conv2D(16, 3, activation=act_f, input_shape=self.state_size)(inp)
            x = layers.Conv2D(16, 3, activation=act_f, input_shape=self.state_size)(x)
            x = layers.MaxPooling2D(3)(x)
            x = layers.Flatten()(x)
            x = layers.Dense(60, activation=act_f)(x)
        else:
            inp = layers.Input((self.state_size,), name="input")
            x = layers.Dense(90, activation=act_f)(inp)

        h0 = layers.Dense(6, activation=act_f)(x)
        output0 = layers.Dense(self.act_ranges[1]-self.act_ranges[0], activation=activations.linear, name="output0")(h0)
        output1 = layers.Dense(self.act_ranges[2]-self.act_ranges[1], activation=activations.linear, name="output1")(x)
        output2 = layers.Dense(self.act_ranges[3]-self.act_ranges[2], activation=activations.linear, name="output2")(x)

        model = models.Model(inputs=inp, outputs=[output0, output1, output2])
        model.compile(optimizers.Adam(self.learning_rate), losses.MeanSquaredError(), metrics=['mse'])
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
        # current_qs_list = tf.Variable(self.model(current_states)).numpy()
        current_qs_list = self.model(current_states)
        temp = [x.numpy() for x in current_qs_list]
        current_qs_list = temp

        # array of states after step from the minibatch
        next_current_states = np.array([transition[3] for transition in minibatch])
        next_current_states = np.squeeze(next_current_states)

        # array of Q-values for our next states
        # next_qs_list = tf.Variable(self.model(next_current_states)).numpy()
        # targ_qs_list = tf.Variable(self.target_model(next_current_states)).numpy()
        next_qs_list = self.model(next_current_states)
        temp = [x.numpy() for x in next_qs_list]
        next_qs_list = temp
        targ_qs_list = self.target_model(next_current_states)
        temp = [x.numpy() for x in targ_qs_list]
        targ_qs_list = temp

        for i, (state, actions, reward, next_state, done) in enumerate(minibatch):
            for j, (action) in enumerate(actions):
                if not done:
                    q_action_list = next_qs_list[j][i]
                    max_next_q = np.argmax(q_action_list)
                    new_q = reward + self.gamma * targ_qs_list[j][i][max_next_q]
                else:
                    new_q = reward

                current_qs_list[j][i][action] = new_q

        np_inputs = current_states
        np_targets = current_qs_list

        history = self.model.fit(
            {"input": np_inputs}, 
            {"output0": np_targets[0], "output1": np_targets[1], "output2": np_targets[2]}, 
            batch_size=self.batch_size,
            verbose=0)

        self.loss = history.history['loss'][0]
        self.mse = (history.history['output1_mse'][0] + history.history['output2_mse'][0]) / 2

        # Increase the target update counter every step
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
        if is_testing:
            return self.predict(state, excluded_actions)

        else:
            if np.random.rand() <= self.epsilon_greedy:
                random_action = []

                for i in range(len(self.action_wrapper.multi_output_ranges) - 1):
                    random_action.append(random.randint(0, self.action_wrapper.multi_output_ranges[i+1] - self.action_wrapper.multi_output_ranges[i]-1))

                # for i in range(len(self.action_wrapper.multi_output_ranges) - 1):
                #     random_action.append(random.choice(self.actions[
                #                                        self.action_wrapper.multi_output_ranges[i]:
                #                                        self.action_wrapper.multi_output_ranges[
                #                                            i + 1]]))

                return random_action
            else:
                return self.predict(state, excluded_actions)

    def predict(self, state, excluded_actions=[]):
        """
        model.predict returns an array of arrays, containing the Q-Values for the actions.
        This function should return the corresponding action with the highest Q-Value.
        """
        #state = np.expand_dims(state, 0)
        q_values = self.model(state)

        return [np.argmax(x) for x in q_values]

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
        if self.per_episode_epsilon_decay and episode >= self.epsilon_decay_ep_start:
            self.decay_epsilon()

        if episode > self.learning_rate_decay_ep_cutoff and self.learning_rate_decay != 1:
            self.decay_lr()

    def ep_reset(self, episode=0):
        """
        This method is mainly used to enact the decay_epsilon and decay_lr
        at the end of every episode.
        """
        if self.per_episode_epsilon_decay and episode >= self.epsilon_decay_ep_start:
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