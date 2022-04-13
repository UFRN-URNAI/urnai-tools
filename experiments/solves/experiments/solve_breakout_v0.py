import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

from absl import app
from tensorflow.keras import layers, models, optimizers
from urnai.agents.generic_agent import GenericAgent
from urnai.agents.rewards.default import PureReward
from urnai.agents.states.gym import GymState
from urnai.envs.gym import GymEnv
from urnai.models.algorithms.dql import DeepQLearning
from urnai.models.memory_representations.neural_network.keras import KerasDeepNeuralNetwork
from urnai.models.model_builder import ModelBuilder
from urnai.trainers.trainer import Trainer




class CustomKerasConvClass(KerasDeepNeuralNetwork):
    def __init__(self, action_output_size, state_input_shape, build_model, gamma, alpha, seed=None,
                 batch_size=32):
        super().__init__(action_output_size, state_input_shape, build_model, gamma, alpha, seed,
                         batch_size)

    def make_model(self):
        self.model = models.Sequential()

        # input layer
        self.model.add(layers.Conv2D(4, 2, input_shape=self.state_input_shape))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.Dropout(0.2))

        # maxpooling
        self.model.add(layers.MaxPooling2D(pool_size=4))
        self.model.add(layers.Dropout(0.2))

        self.model.add(layers.Flatten())

        # hidden 8 neuron
        self.model.add(layers.Dense(8))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())

        # output layer (4 neuron output)
        self.model.add(layers.Dense(self.action_output_size))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation('linear'))

        self.model.compile(optimizer=optimizers.Adam(lr=self.alpha), loss='mse',
                           metrics=['accuracy'])


def declare_trainer():
    env = GymEnv(id='Breakout-v0', render=True)
    env.reset()

    action_wrapper = env.get_action_wrapper()
    state_builder = GymState(env.env_instance.observation_space.shape)

    helper = ModelBuilder()
    helper.add_convolutional_layer(filters=8, kernel_size=4,
                                   input_shape=env.env_instance.observation_space.shape)
    helper.add_maxpooling_layer()
    helper.add_flatten_layer()
    helper.add_fullyconn_layer(10)
    helper.add_output_layer()

    dq_network = DeepQLearning(action_wrapper=action_wrapper, state_builder=state_builder,
                               build_model=helper.get_model_layout(),
                               gamma=0.99, learning_rate=0.01, epsilon_decay=0.9999,
                               epsilon_min=0.005, memory_maxlen=50000, min_memory_size=2000,
                               lib='keras')

    agent = GenericAgent(dq_network, PureReward())

    trainer = Trainer(env, agent, save_path='urnai/models/saved', file_name='breakout-v0_keras',
                      save_every=100, enable_save=True, relative_path=True,
                      max_training_episodes=50, max_steps_training=800,
                      max_test_episodes=5, max_steps_testing=800)
    return trainer


def main(unused_argv):
    try:
        trainer = declare_trainer()
        trainer.train()
        trainer.play()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
