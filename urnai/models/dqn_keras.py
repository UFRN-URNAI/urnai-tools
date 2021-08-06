import tensorflow as tf
import numpy as np
import random
import os, sys
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from .base.abmodel import LearningModel
from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import StateBuilder
from .model_builder import ModelBuilder
from urnai.utils.error import IncoherentBuildModelError
from urnai.utils.error import UnsupportedBuildModelLayerTypeError

class DQNKeras(LearningModel):

    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, gamma=0.99, 
                learning_rate=0.001, learning_rate_min = 0.0001, learning_rate_decay = 0.99995, learning_rate_decay_ep_cutoff = 0,
                name='DQN', epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=32, batch_training=False,
                memory_maxlen=50000, use_memory=True, per_episode_epsilon_decay=False, build_model = ModelBuilder.DEFAULT_BUILD_MODEL,
                seed_value=None, cpu_only=False, epsilon_linear_decay=False, lr_linear_decay=False):
        
        super(DQNKeras, self).__init__(action_wrapper, state_builder, gamma, learning_rate, learning_rate_min, learning_rate_decay, 
                                        epsilon_start, epsilon_min, epsilon_decay, per_episode_epsilon_decay, learning_rate_decay_ep_cutoff, name, seed_value, cpu_only, epsilon_linear_decay, lr_linear_decay)
        self.batch_size = batch_size
        self.batch_training = batch_training

        self.build_model = build_model
        self.model = self.make_model()
        self.use_memory = use_memory

        if self.use_memory:
            self.memory = deque(maxlen=memory_maxlen)
            self.memory_maxlen = memory_maxlen
        
    def make_model(self):
        model = Sequential()

        if self.build_model[0]['type'] == ModelBuilder.LAYER_INPUT and self.build_model[-1]['type'] == ModelBuilder.LAYER_OUTPUT:
            self.build_model[0]['shape'] = [None, self.state_size]
        
        self.build_model[-1]['length'] = self.action_size

        for idx, (layer_model) in enumerate(self.build_model):
            if layer_model['type'] == ModelBuilder.LAYER_INPUT: 
                if self.build_model.index(layer_model) == 0:
                    model.add(Dense(layer_model['nodes'], input_dim=layer_model['shape'][1], activation='relu'))
                else:
                    raise IncoherentBuildModelError("Input Layer must be the first one.") 

            elif layer_model['type'] == ModelBuilder.LAYER_FULLY_CONNECTED:
                # if previous layer is convolutional, add a Flatten layer before the fully connected
                if self.build_model[idx]['type'] == ModelBuilder.LAYER_CONVOLUTIONAL:
                    model.add(Flatten())

                model.add(Dense(layer_model['nodes'], activation='relu'))

            elif layer_model['type'] == ModelBuilder.LAYER_OUTPUT:
                # if previous layer is convolutional, add a Flatten layer before the fully connected
                if self.build_model[idx]['type'] == ModelBuilder.LAYER_CONVOLUTIONAL:
                    model.add(Flatten())

                model.add(Dense(layer_model['length'], activation='linear'))

            elif layer_model['type'] == ModelBuilder.LAYER_CONVOLUTIONAL:
                # if convolutional layer is the first, it's going to have the input shape and be treated as the input layer
                if self.build_model.index(layer_model) == 0:
                    model.add(Conv2D(layer_model['filters'], layer_model['filter_shape'], 
                              padding=layer_model['padding'], activation='relu', input_shape=layer_model['input_shape']))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=layer_model['max_pooling_pool_size_shape']))
                    try:
                        #this code makes dropout layer optional
                        model.add(Dropout(layer_model['dropout']))
                    except KeyError:
                        pass
                else:
                    model.add(Conv2D(layer_model['filters'], layer_model['filter_shape'], 
                              padding=layer_model['padding'], activation='relu'))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=layer_model['max_pooling_pool_size_shape']))
                    try:
                        #this code makes dropout layer optional
                        model.add(Dropout(layer_model['dropout']))
                    except KeyError:
                        pass
            else:
                raise UnsupportedBuildModelLayerTypeError("Unsuported Layer Type " + layer_model['type'])

        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse', metrics=['accuracy'])
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        if not hasattr(self, 'batch_training') or not self.batch_training:
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=self.tensorboard_callback)
        else:
            inputs = np.zeros((len(minibatch), self.state_size))
            targets = np.zeros((len(minibatch), self.action_size))
            i=0

            for state, action, reward, next_state, done in minibatch:
                q_current_state = self.model.predict(state)[0]
                q_next_state = self.model.predict(next_state)[0]

                inputs[i] = state
                targets[i] = q_current_state

                if done:
                    targets[i,np.argmax(action)] = reward
                else:
                    targets[i,np.argmax(action)] = reward + self.gamma * np.max(q_next_state)

                i+=1

            loss = self.model.train_on_batch(inputs, targets)

        #Epsilon decay operation was here, moved it to "decay_epsilon()" and to "learn()"

    def no_memory_learning(self, s, a, r, s_, done):
            target = r 
            if not done:
                target = (r + self.gamma * np.amax(self.model.predict(s_)[0]))
            target_f = self.model.predict(s)
            target_f[0][a] = target
            self.model.fit(s, target_f, epochs=1, verbose=0, callbacks=self.tensorboard_callback)

    def learn(self, s, a, r, s_, done):
        if self.use_memory:
            self.memorize(s, a, r, s_, done)
            if(len(self.memory) > self.batch_size):
                self.replay()
        else:
            self.no_memory_learning(s, a, r, s_, done)

        if not self.per_episode_epsilon_decay:
            self.decay_epsilon()

    def choose_action(self, state, excluded_actions=[], is_testing=False):
        # Verifies if we are running a test (Evaluating our agent)
        if is_testing:
            return self.predict(state, excluded_actions)
        # If we are not testing (therefore we are training), evaluate epsilon greedy strategy
        else:
            if np.random.rand() <= self.epsilon_greedy:
                random_action = random.choice(self.actions)
                # Removing excluded actions
                while random_action in excluded_actions:
                    random_action = random.choice(self.actions)
                return random_action
            else:
                return self.predict(state, excluded_actions)
        
    def predict(self, state, excluded_actions=[]):
        '''
        model.predict returns an array of arrays, containing the Q-Values for the actions. This function should return the
        corresponding action with the highest Q-Value.
        '''
        q_values = self.model.predict(state)[0]
        action_idx = int(np.argmax(q_values))

        while action_idx in excluded_actions:
            q_values = np.delete(q_values, action_idx)
            action_idx = int(np.argmax(q_values))

        return action_idx

    def save_extra(self, persist_path):
        self.model.save_weights(self.get_full_persistance_path(persist_path)+".h5")

    def load_extra(self, persist_path):
        exists = os.path.isfile(self.get_full_persistance_path(persist_path)+".h5")

        if(exists):
            self.model = self.make_model()
            self.model.load_weights(self.get_full_persistance_path(persist_path)+".h5")
            
            self.set_seeds()
