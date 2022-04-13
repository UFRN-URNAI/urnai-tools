from datetime import datetime
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from absl import app
from urnai.agents.generic_agent import GenericAgent
from urnai.agents.rewards.gym import FrozenlakeReward
from urnai.agents.states.gym import FrozenLakeState
from urnai.envs.gym import GymEnv
from urnai.models.ddqn_keras import DDQNKeras
from urnai.models.model_builder import ModelBuilder
from urnai.trainers.trainer import Trainer


def declare_trainer():
    env = GymEnv(id='FrozenLakeNotSlippery-v0')

    action_wrapper = env.get_action_wrapper()
    state_builder = FrozenLakeState()

    training_date = str(datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')

    helper = ModelBuilder()
    helper.add_input_layer()
    helper.add_fullyconn_layer(256)
    helper.add_fullyconn_layer(256)
    helper.add_fullyconn_layer(256)
    helper.add_fullyconn_layer(256)
    helper.add_output_layer()
    dq_network = DDQNKeras(action_wrapper=action_wrapper, state_builder=state_builder,
                           learning_rate=0.005, gamma=0.90, use_memory=False,
                           per_episode_epsilon_decay=True, build_model=helper.get_model_layout())
    agent = GenericAgent(dq_network, FrozenlakeReward())
    trainer = Trainer(env, agent,
                      file_name=training_date + os.path.sep + 'frozenlake_test_ddqnKeras',
                      save_every=1000, enable_save=True,
                      max_training_episodes=1000, max_steps_training=500,
                      max_test_episodes=10, max_steps_testing=500)
    return trainer


def main(unused_argv):
    try:
        # FrozenLake is solved when the agent is able to reach the end of the maze 100% of the times
        trainer = declare_trainer()
        trainer.train()
        trainer.play()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
