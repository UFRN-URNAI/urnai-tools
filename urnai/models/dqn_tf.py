import tensorflow as tf
from tensorflow.python.framework import ops
from urnai.utils.error import IncoherentBuildModelError, UnsupportedBuildModelLayerTypeError, DeprecatedCodeException
from tensorflow.compat.v1 import Session,ConfigProto,placeholder,layers,train,global_variables_initializer
import numpy as np
import random
import os
import pickle
from collections import deque
from .base.abmodel import LearningModel
from urnai.agents.actions.base.abwrapper import ActionWrapper
from urnai.agents.states.abstate import StateBuilder
from .model_builder import ModelBuilder

class DqlTensorFlow(LearningModel):

    #by default learning rate should not decay at all, since this is not the default behavior
    #of Deep-Q Learning
    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, learning_rate=0.0002, learning_rate_min=0.00002, learning_rate_decay=1, learning_rate_decay_ep_cutoff=0, gamma=0.95, name='DQN', build_model = ModelBuilder.DEFAULT_BUILD_MODEL, epsilon_start=1.0, epsilon_min=0.5, epsilon_decay=0.995, per_episode_epsilon_decay=False, use_memory=False, memory_maxlen=10000, batch_training=False, batch_size=32, min_memory_size=5000, seed_value=None, cpu_only=False, epsilon_linear_decay=False, lr_linear_decay=False):
        super().__init__(action_wrapper, state_builder, gamma, learning_rate, learning_rate_min, learning_rate_decay, epsilon_start, epsilon_min, epsilon_decay , per_episode_epsilon_decay, learning_rate_decay_ep_cutoff, name, seed_value, cpu_only, epsilon_linear_decay, lr_linear_decay)
        # Defining the model's layers. Tensorflow's objects are stored into self.model_layers
        self.batch_size = batch_size
        self.build_model = build_model
        self.make_model()

        self.use_memory = use_memory
        if self.use_memory:
            self.memory = deque(maxlen=memory_maxlen)
            self.memory_maxlen = memory_maxlen
            self.min_memory_size = min_memory_size

    def learn(self, s, a, r, s_, done):
        if self.use_memory:
            self.memory_learn(s, a, r, s_, done)
        else:
            self.no_memory_learn(s, a, r, s_, done)

    def memory_learn(self, s, a, r, s_, done):
        self.memorize(s, a, r, s_, done)
        if len(self.memory) < self.min_memory_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = np.array([val[0] for val in batch])
        states = np.squeeze(states)
        next_states = np.array([(np.zeros(self.state_size)
                                if val[3] is None else val[3]) for val in batch])
        next_states = np.squeeze(next_states)
        # predict Q(s,a) given the batch of states
        q_s_a = self.sess.run(self.model_layers[-1], feed_dict={self.model_layers[0]: states})

        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self.sess.run(self.model_layers[-1], feed_dict={self.model_layers[0]: next_states})

        # setup training arrays
        x = np.zeros((len(batch), self.state_size))
        y = np.zeros((len(batch), self.action_size))

        for i, (state, action, reward, next_state, done) in enumerate(batch):
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            if done:
                # if this is the last step, there is no future max q value, so we the new_q is just the reward
                current_q[action] = reward
            else:
                # new Q-value is equal to the reward at that step + discount factor * the max q-value for the next_state
                current_q[action] = reward + self.gamma * np.amax(q_s_a_d[i])
            
            x[i] = state
            y[i] = current_q

        self.sess.run(self.optimizer, feed_dict={self.model_layers[0]: x, self.tf_qsa: y})

    def no_memory_learn(self, s, a, r, s_, done):
        qsa_values = self.sess.run(self.model_layers[-1], feed_dict={self.model_layers[0]: s})

        current_q = 0

        if done:
            current_q = r
        else:
            current_q = r + self.gamma * self.__maxq(s_)

        qsa_values[0, a] = current_q

        self.sess.run(self.optimizer, feed_dict={self.model_layers[0] : s, self.tf_qsa: qsa_values})

    def __maxq(self, state):
        values = self.sess.run(self.model_layers[-1], feed_dict={self.model_layers[0]: state})

        index = np.argmax(values[0])
        mxq = values[0, index]

        return mxq

    def choose_action(self, state, excluded_actions=[], is_testing=False):
        if is_testing:
            return self.predict(state, excluded_actions)
        
        else:
            expl_expt_tradeoff = np.random.rand()
            action = None

            if self.epsilon_greedy > expl_expt_tradeoff:
                ex_int = self.actions[random.randint(0, len(self.actions) - 1)]
                random_action = random.choice(self.actions)

                # Removing excluded actions
                while random_action in excluded_actions:
                    random_action = random.choice(self.actions)
                action = random_action
            else:
                action = self.predict(state, excluded_actions)

            if not self.per_episode_epsilon_decay:
                self.decay_epsilon()

            return action

    def predict(self, state, excluded_actions=[]):
        q_values = self.sess.run(self.model_layers[-1], feed_dict={self.model_layers[0]: state})
        action_idx = np.argmax(q_values)

        # Removing excluded actions
        # This is possibly badly optimized, eventually look back into this
        while action_idx in excluded_actions:
            q_values = np.delete(q_values, action_idx)
            action_idx = np.argmax(q_values)
        
        action = int(action_idx)
        return action

    def save_extra(self, persist_path):
        #Saving tensorflow stuff
        self.saver.save(self.sess, self.get_full_persistance_tensorflow_path(persist_path))

    def load_extra(self, persist_path):
        #Makes model, needed to be done before loading tensorflow's persistance
        self.make_model()
        #Check if tf file exists
        exists = os.path.isfile(self.get_full_persistance_tensorflow_path(persist_path) + ".meta")
        #If yes, load it
        if exists:
            self.saver.restore(self.sess, self.get_full_persistance_tensorflow_path(persist_path))
            self.set_seed()

    def make_model(self):
        #These are already inside make_model(), commenting out
        ops.reset_default_graph()
        tf.compat.v1.disable_eager_execution()

        # Initializing TensorFlow session
        self.sess = Session(config=ConfigProto(allow_soft_placement=True))

        if self.build_model[0]['type'] == ModelBuilder.LAYER_INPUT and self.build_model[-1]['type'] == ModelBuilder.LAYER_OUTPUT:
            self.build_model[0]['shape'] = [None, self.state_size]
            self.build_model[-1]['length'] = self.action_size

        #Load each layer
        self.model_layers = []
        for layer_model in self.build_model:
            if layer_model['type'] == ModelBuilder.LAYER_INPUT: 
                if self.build_model.index(layer_model) == 0:
                    self.model_layers.append(placeholder(dtype=tf.float32, 
                        shape=layer_model['shape'], name='inputs_'))
                else:
                    raise IncoherentBuildModelError("Input Layer must be the first one.") 
            elif layer_model['type'] == ModelBuilder.LAYER_FULLY_CONNECTED:
                self.model_layers.append(layers.dense(inputs=self.model_layers[-1], 
                    units=layer_model['nodes'], 
                    activation=tf.nn.relu, name=layer_model['name']))
            elif layer_model['type'] == ModelBuilder.LAYER_OUTPUT:
                self.model_layers.append(layers.dense(inputs=self.model_layers[-1], 
                    units=self.action_size,activation=None))
            else:
                raise UnsupportedBuildModelLayerTypeError("Unsuported Layer Type " + layer_model['type'])

        #Setup output qsa layer and loss
        self.tf_qsa = placeholder(shape=[None, self.action_size], dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(self.tf_qsa, self.model_layers[-1])
        self.optimizer = train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        #self.logits = layers.dense(self.model_layers[-1], self.action_size)
        #self._states = placeholder(shape=[None, self.state_size], dtype=tf.float32)

        self.sess.run(global_variables_initializer())

        self.saver = train.Saver()

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

