import pathlib
import sys

from urnai.agents.rewards.default import PureReward
from urnai.agents.states.gym import GymState
from urnai.envs.gym import GymEnv
from urnai.models.ddqn_keras import DDQNKeras
from urnai.models.model_builder import ModelBuilder
from urnai.agents.generic_agent import GenericAgent
from urnai.trainers.trainer import Trainer
#     solve_simple64_veryeasy
# from urnai.solves.experiments import solve_breakout_ram_v0, solve_breakout_v0, \
#     solve_cartpole_v1_dql, solve_simple64, solve_simple64_mo

def test_cartpole_v1_declaration():
    env = GymEnv(id='CartPole-v1')

    action_wrapper = env.get_action_wrapper()
    state_builder = GymState(env.env_instance.observation_space.shape[0])

    helper = ModelBuilder()
    helper.add_input_layer(nodes=25)
    helper.add_fullyconn_layer(25)
    helper.add_output_layer()

    dq_network = DDQNKeras(action_wrapper=action_wrapper, state_builder=state_builder,
                           build_model=helper.get_model_layout(), use_memory=False,
                           gamma=0.99, learning_rate=0.001, epsilon_decay=0.9997, epsilon_min=0.01,
                           memory_maxlen=50000, min_memory_size=1000)

    agent = GenericAgent(dq_network, PureReward())

    trainer = Trainer(env, agent, save_path='urnai/models/saved',
                      file_name='cartpole_v1_ddqn_25x25_test',
                      save_every=100, enable_save=True, relative_path=True,
                      max_training_episodes=1000, max_steps_training=520,
                      max_test_episodes=100, max_steps_testing=520)
