import numpy
import random
from collections import deque
from models.memory_representations.neural_network.pytorch import PyTorchDeepNeuralNetwork as DeepNeuralNetwork
from models.base.abmodel import LearningModel
from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import StateBuilder

class DeepQLearning(LearningModel):
    #by default learning rate should not decay at all, since this is not the default behavior
    #of Deep-Q Learning
    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, learning_rate=0.0002, learning_rate_min=0.00002, learning_rate_decay=1, learning_rate_decay_ep_cutoff=0, gamma=0.95, name='DQN', build_model = None, epsilon_start=1.0, epsilon_min=0.5, epsilon_decay=0.995, per_episode_epsilon_decay=False, use_memory=False, memory_maxlen=10000, batch_size=32, min_memory_size=5000, seed_value=None, cpu_only=False):
        super().__init__(action_wrapper, state_builder, gamma, learning_rate, learning_rate_min, learning_rate_decay, epsilon_start, epsilon_min, epsilon_decay , per_episode_epsilon_decay, learning_rate_decay_ep_cutoff, name, seed_value, cpu_only)
        # Defining the model's layers. Tensorflow's objects are stored into self.model_layers
        self.batch_size = batch_size
        self.build_model = build_model
        self.dnn = DeepNeuralNetwork(self.action_size, self.state_size, self.build_model, self.gamma, self.learning_rate, self.seed_value) 

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
        states = numpy.array([val[0] for val in batch])
        states = numpy.squeeze(states)
        next_states = numpy.array([(numpy.zeros(self.state_size)
                                if val[3] is None else val[3]) for val in batch])
        next_states = numpy.squeeze(next_states)

        # predict Q(s,a) given the batch of states
        rows = self.batch_size
        cols = self.action_size
        q_s_a = numpy.zeros(shape=(rows, cols))
        
        #TODO: Try and implement this generically for different Libraries 
        # (tf, keras and pytorch all probably have some built-in way of batch inference)
        for i in range(self.batch_size):
            q_s_a[i] = self.dnn.get_output(state)

        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        rows = self.batch_size
        cols = self.action_size
        q_s_a_d = numpy.zeros(shape=(rows, cols))
        
        #TODO: Try and implement this generically for different Libraries 
        # (tf, keras and pytorch all probably have some built-in way of batch inference)
        for i in range(self.batch_size):
            q_s_a_d[i] = self.dnn.get_output(state)

        # setup training arrays
        target_q_values = numpy.zeros((len(batch), self.action_size))

        for i, (state, action, reward, next_state, done) in enumerate(batch):
            # get the current q values for all actions in state
            current_q = numpy.copy(q_s_a[i])
            if done:
                # if this is the last step, there is no future max q value, so we the new_q is just the reward
                current_q[action] = reward
            else:
                # new Q-value is equal to the reward at that step + discount factor * the max q-value for the next_state
                current_q[action] = reward + self.gamma * numpy.amax(q_s_a_d[i])
            
            target_q_values[i] = current_q

        #update neural network with expected q values
        for i in range(self.batch_size):
            self.dnn.update(states[i], target_q_values[i])

    def no_memory_learn(self, s, a, r, s_, done):
        #get output for current sars array
        rows = 1 
        cols = self.action_size
        target_q_values = numpy.zeros(shape=(rows, cols))

        expected_q = 0
        if done:
            expected_q = r
        else:
            expected_q = r + self.gamma * self.__maxq(s_)

        target_q_values[0, a] = expected_q 

        self.dnn.update(s, target_q_values)

    def __maxq(self, state):
        values = self.dnn.get_output(state)
        mxq = values.max()
        return mxq

    def choose_action(self, state, excluded_actions=[]):
        expl_expt_tradeoff = numpy.random.rand()

        if self.epsilon_greedy > expl_expt_tradeoff:
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
        q_values = self.dnn.get_output(state)
        action_idx = numpy.argmax(q_values)

        # Removing excluded actions
        # TODO: This is possibly badly optimized, eventually look back into this
        while action_idx in excluded_actions:
            q_values = numpy.delete(q_values, action_idx)
            action_idx = numpy.argmax(q_values)
        
        action = int(action_idx)
        return action

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save_extra(self, persist_path):
        self.dnn.save(persist_path)

    def load_extra(self, persist_path):
        self.dnn.load(persist_path)
