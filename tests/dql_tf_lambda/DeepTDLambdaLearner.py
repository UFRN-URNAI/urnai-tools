import random

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()


class DeepTDLambdaLearner:
    def __init__(self,
                 n_actions,
                 n_states,
                 discount=0.8,
                 alpha=0.1,
                 epsilon=1,
                 epsilon_decay=0.99,
                 lamb=0.5,
                 ):
        self.n_actions = n_actions
        self.n_states = n_states
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.lamb = lamb

        tf.compat.v1.reset_default_graph
        self.state_tensor, self.Q_values_tensor, self.chosen_value_tensor, self.opt, \
            self.weight1 = self._build_model()
        self.grads_and_vars = self._get_gradients(self.opt)
        self.e_trace = self._get_eligibility_trace(self.grads_and_vars)
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    # The model runs faster when a bias layer isn't used, and in fact seems to perform better.
    def _build_model(self):
        state_tensor = tf.compat.v1.placeholder(tf.float32, shape=(1, self.n_states))
        weight1 = tf.Variable(tf.zeros(shape=(self.n_states, self.n_actions)))
        # bias1 = tf.Variable(tf.zeros(self.n_actions))
        # Q_values_tensor = tf.add(tf.matmul(state_tensor, weight1), bias1)
        Q_values_tensor = tf.matmul(state_tensor, weight1)
        chosen_action_index = tf.argmax(Q_values_tensor, 1)
        chosen_value_tensor = tf.gather(Q_values_tensor, chosen_action_index, axis=1)
        opt = tf.compat.v1.train.GradientDescentOptimizer(self.alpha)

        return state_tensor, Q_values_tensor, chosen_value_tensor, opt, weight1

    def _get_gradients(self, opt):
        trainable_variables = tf.compat.v1.trainable_variables()
        grads_and_vars = opt.compute_gradients(self.chosen_value_tensor, trainable_variables)
        return grads_and_vars

    def _get_eligibility_trace(self, grads_and_vars):
        e_trace = []
        for gv in grads_and_vars:
            e = np.zeros(gv[0].get_shape())
            e_trace.append(e)
        return e_trace

    def _compute_e_trace(self, evaluated_gradients, e_trace):
        for i in range(len(e_trace)):
            e_trace[i] = self.discount * self.lamb * e_trace[i] + evaluated_gradients[i]
            assert (e_trace[i].shape == evaluated_gradients[i].shape)
        return e_trace

    def predict_Q_values(self, state):
        Q_values = self.sess.run(self.Q_values_tensor, feed_dict={self.state_tensor: state})
        return Q_values

    def get_max_Q_value(self, state):
        Q_values = self.predict_Q_values(state)
        return np.max(Q_values)

    def get_Q_value(self, state, action):
        Q_values = self.predict_Q_values(state)
        return Q_values[0][action]

    def get_best_action(self, state):
        Q_values = self.predict_Q_values(state)
        return np.argmax(Q_values)

    def get_e_greedy_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions), True
        else:
            return self.get_best_action(state), False

    def reset_e_trace(self):
        self.e_trace = [0 * e for e in self.e_trace]

    def print_weights(self):
        w1 = self.sess.run(self.weight1)
        print(w1)

    def print_Q_values(self, state):
        print(self.predict_Q_values(state))

    # The most important function
    def learn(self, state, action, next_state, reward, greedy):
        target = reward + self.get_max_Q_value(next_state)
        old_Q = self.get_Q_value(state, action)
        delta = target - old_Q

        # For debugging purposes. Initially had problems with numbers getting too large.
        if abs(delta) > 1000:
            print('Warning! Delta getting very big. Delta = {}'.format(delta))

        if greedy:
            self.reset_e_trace()
        else:
            grads_and_vars = self.sess.run(self.grads_and_vars,
                                           feed_dict={self.state_tensor: state})
            evaluated_gradients = [gv[0] for gv in grads_and_vars]
            self.e_trace = self._compute_e_trace(evaluated_gradients, self.e_trace)

        change = [-delta * e for e in self.e_trace]

        grad_placeholder = [(tf.compat.v1.placeholder('float', shape=grad[0].get_shape()), grad[1])
                            for grad in self.grads_and_vars]
        apply_placeholder_op = self.opt.apply_gradients(grad_placeholder)

        feed_dict = {}
        for i in range(len(grad_placeholder)):
            feed_dict[grad_placeholder[i][0]] = change[i]
        self.sess.run(apply_placeholder_op, feed_dict=feed_dict)

        # Decay epsilon
        self.epsilon = self.epsilon * self.epsilon_decay
