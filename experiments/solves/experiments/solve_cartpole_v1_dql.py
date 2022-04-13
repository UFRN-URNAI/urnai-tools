import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

from absl import app
from urnai.agents.generic_agent import GenericAgent
from urnai.agents.rewards.default import PureReward
from urnai.agents.states.gym import GymState
from urnai.envs.gym import GymEnv
from urnai.models.algorithms.dql import DeepQLearning
from urnai.models.algorithms.dql_lambda import DeepQLearningLambda
from urnai.models.model_builder import ModelBuilder
from urnai.trainers.trainer import Trainer


def declare_trainer():
    env = GymEnv(id='CartPole-v1', render=False)

    action_wrapper = env.get_action_wrapper()
    state_builder = GymState(env.env_instance.observation_space.shape[0])

    helper = ModelBuilder()
    helper.add_input_layer(nodes=50)
    # helper.add_fullyconn_layer(50)
    helper.add_output_layer()

    dq_network = DeepQLearning(action_wrapper=action_wrapper, state_builder=state_builder,
                               build_model=helper.get_model_layout(), gamma=0.99, use_memory=True,
                               learning_rate=0.001, epsilon_decay=0.9997, epsilon_min=0.01,
                               memory_maxlen=50000, min_memory_size=100, batch_size=32,
                               lib='pytorch')

    dq_network = DeepQLearningLambda(action_wrapper=action_wrapper, state_builder=state_builder,
                                     build_model=helper.get_model_layout(),
                                     lamb=0.9, gamma=0.99, learning_rate=0.001, memory_maxlen=128,
                                     min_memory_size=64, lib='keras_e_traces',
                                     epsilon_decay=0.99999, epsilon_start=1.0, epsilon_min=0.005,
                                     epsilon_linear_decay=False, per_episode_epsilon_decay=False)

    agent = GenericAgent(dq_network, PureReward())

    # Cartpole-v1 is solved when avg. reward over 100 episodes is greater than or equal to 475
    trainer = Trainer(env, agent, save_path='urnai/models/saved',
                      file_name='cartpole_v1_dql_pytorch1',
                      save_every=100, enable_save=True, relative_path=True,
                      max_training_episodes=1000, max_steps_training=1000,
                      max_test_episodes=100, max_steps_testing=1000)
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
