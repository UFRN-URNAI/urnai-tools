import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from absl import app
from urnai.agents.generic_agent import GenericAgent
from urnai.agents.rewards.default import PureReward
from urnai.agents.states.gym import GymState
from urnai.envs.gym import GymEnv
from urnai.models.ddqn_keras import DDQNKeras
from urnai.models.model_builder import ModelBuilder
from urnai.trainers.trainer import Trainer


def declare_trainer():
    env = GymEnv(id='CartPole-v0')

    action_wrapper = env.get_action_wrapper()
    state_builder = GymState(env.env_instance.observation_space.shape[0])

    helper = ModelBuilder()
    helper.add_input_layer()
    helper.add_fullyconn_layer(25)
    helper.add_output_layer()

    dq_network = DDQNKeras(action_wrapper=action_wrapper, state_builder=state_builder,
                           build_model=helper.get_model_layout(), use_memory=False,
                           gamma=0.99, learning_rate=0.001, epsilon_decay=0.9997, epsilon_min=0.01,
                           memory_maxlen=50000, min_memory_size=1000)

    agent = GenericAgent(dq_network, PureReward())

    # Cartpole-v0 is solved when avg. reward over 100 episodes is greater than or equal to 195
    trainer = Trainer(env, agent, save_path='urnai/models/saved',
                      file_name='cartpole-v0-unified-step-play3',
                      save_every=100, enable_save=True, relative_path=True,
                      max_training_episodes=1000, max_steps_training=500,
                      max_test_episodes=100, max_steps_testing=500)
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
