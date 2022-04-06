from abc import abstractmethod

from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import StateBuilder
from base.savable import Savable
from utils.returns import ActionIndex


class LearningModel(Savable):
    """
    Base Class for all Reinforcement Learning Algorithms.

    This base class defines all necessary methods for most RL algorithms,
    such as learn, predict, implementation of epsilon greedy strategy etc.

    Beyond that, it also sets many important internal attributes, such as
    the learning rate, gamma, action size, state size, minimum epsilon value etc.

    Parameters:
        action_wrapper: Object
            Object responsible for describing possible actions
        state_builder: Object
            Object responsible for creating states from the game environment
        gamma: Float
            Gamma parameter in the Deep Q Learning algorithm
        learning_rate: Float
            Rate at which the deep learning algorithm will learn
            (alpha on most mathematical representations)
        learning_rate_min: Float
            Minimum value that learning_rate will reach throught training
        learning_rate_decay: Float
            Inverse of the rate at which the learning rate will decay each episode
            (defaults to 1 so no decay)
        epsilon_start: Float
            Value that the epsilon from epsilon greedy strategy will start from (defaults to 1)
        epsilon_min: Float
            Minimum value that epsilon will reach trough training
        epsilon_decay_rate: Float
            Inverse of the rate at which the epsilon value will decay each step
            (0.99 => 1% will decay each step)
        per_episode_epsilon_decay:  Bool
            Whether or not the epsilon decay will be done each episode, instead of each step
        learning_rate_decay_ep_cutoff: Integer
            Episode at which learning rate decay will start (defaults to 0)
        name: String
            Name of the algorithm implemented
        seed_value: Integer (default None)
            Value to assing to random number generators in Python and our ML libraries to try
            and create reproducible experiments
        cpu_only: Bool
            If true will run algorithm only using CPU, also useful for reproducibility since GPU
            paralelization creates uncertainty
    """

    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, gamma,
                 learning_rate, learning_rate_min, learning_rate_decay,
                 epsilon_start, epsilon_min, epsilon_decay_rate, per_episode_epsilon_decay=False,
                 learning_rate_decay_ep_cutoff=0,
                 name=None, seed_value=None, cpu_only=False, epsilon_linear_decay=False,
                 lr_linear_decay=False):
        super(LearningModel, self).__init__()

        self.seed_value = seed_value
        self.cpu_only = cpu_only
        self.set_seeds()

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learning_rate_min = learning_rate_min
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_ep_cutoff = learning_rate_decay_ep_cutoff
        self.lr_linear_decay = lr_linear_decay

        self.name = name
        self.action_wrapper = action_wrapper
        self.state_builder = state_builder
        self.actions = action_wrapper.get_actions()
        self.action_size = action_wrapper.get_action_space_dim()
        self.state_size = state_builder.get_state_dim()

        # EXPLORATION PARAMETERS FOR EPSILON GREEDY STRATEGY
        self.epsilon_greedy = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = epsilon_decay_rate
        self.per_episode_epsilon_decay = per_episode_epsilon_decay
        self.epsilon_linear_decay = epsilon_linear_decay

        # self.tensorboard_callback_logdir = ''
        self.tensorboard_callback = None

    @abstractmethod
    def learn(self, s, a, r, s_, done, is_last_step: bool) -> None:
        """
        Implements the learning strategy of the specific reinforcement learning
        algorithm implemented in the classes that inherit from LearningModel.
        """
        ...

    @abstractmethod
    def choose_action(self, state, excluded_actions=[], is_testing=False) -> ActionIndex:
        """
        Implements the logic for choosing an action while training and while testing an Agent.
        For most Reinforcement Learning Algorithms, this method will choose an action directly
        When is_testing=True, and will implement the exploration algorithm for when
        is_testing=False.
        One such exploration algorithm commonly used it the epsilon greedy strategy.
        """
        pass

    @abstractmethod
    def predict(self, state, excluded_actions=[]) -> ActionIndex:
        """Given a State, returns the index for the action with the highest Q-Value."""
        pass

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
            if self.learning_rate > self.learning_rate_min:
                self.learning_rate -= (1 - self.learning_rate_decay)
        else:
            if self.learning_rate > self.learning_rate_min:
                self.learning_rate *= self.learning_rate_decay

    def ep_reset(self, episode=0):
        """
        This method is mainly used to enact the decay_epsilon and decay_lr
        at the end of every episode.
        """
        if self.per_episode_epsilon_decay:
            self.decay_epsilon()

        if episode > self.learning_rate_decay_ep_cutoff and self.learning_rate_decay != 1:
            self.decay_lr()

    def set_seeds(self):
        if self.seed_value is not None:
            """
            This method sets seeds for the Random Number Generators of Python and Numpy.
            This is done with the objective of removing non-deterministic calculations in
            the process of training and inference of neural networks, since ML libraries
            such as keras or tensorflow use RNGs from numpy and python in their inerworkings.

            Another functionality of this method is that it forces execution of ML libraries on
            the CPU, since paralelized calculations in GPU lead to non-deterministic operations.
            This is done through the 'cpu_only' parameter, which is independent from seed_value.

            However, for a machine learning model to be fully deterministic we also need the
            RNG inside the library that implements the model to have its seed fixed. This is done
            in a separate method dependent on each ML library that URNAI supports, for example
            inside keras.py there is the set_seet method which will set the tensorflow RNG seed.

            more info on:
            https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
            """
            import os

            if self.cpu_only:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''  # To be used when GPU usage is not desired

            os.environ['PYTHONHASHSEED'] = str(self.seed_value)

            import numpy as np
            np.random.seed(self.seed_value)

            import random as python_random
            python_random.seed(self.seed_value)
