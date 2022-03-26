import pathlib
import sys

from absl import app
from urnai.agents.generic_agent import GenericAgent
from urnai.agents.rewards.default import PureReward
from urnai.agents.states.gym import GymState
from urnai.envs.gym import GymEnv
from urnai.models.algorithms.dql import DeepQLearning
from urnai.models.model_builder import ModelBuilder
from urnai.trainers.trainer import Trainer

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))


def declare_trainer():
    env = GymEnv(id='Breakout-ram-v0', render=True)

    action_wrapper = env.get_action_wrapper()
    state_builder = GymState(env.env_instance.observation_space.shape[0])

    helper = ModelBuilder()
    helper.add_input_layer(nodes=66)
    helper.add_output_layer()

    dq_network = DeepQLearning(action_wrapper=action_wrapper, state_builder=state_builder,
                               build_model=helper.get_model_layout(),
                               gamma=0.99, learning_rate=0.01, epsilon_decay=0.998,
                               epsilon_min=0.01, memory_maxlen=100000, min_memory_size=4000,
                               per_episode_epsilon_decay=True, lib='pytorch')

    agent = GenericAgent(dq_network, PureReward())

    trainer = Trainer(
        env,
        agent,
        save_path='urnai/models/saved',
        file_name='breakout-ram-v0_dql_66_pytorch',
        save_every=300,
        enable_save=True,
        relative_path=True,
        max_training_episodes=1200,
        max_steps_training=1800,
        max_test_episodes=100,
        max_steps_testing=1800,
    )
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
