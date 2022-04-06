import pathlib
import sys

from absl import app
import tensorflow as tf
from urnai.agents.actions.sc2_wrapper import SimpleTerranWrapper
from urnai.agents.rewards.sc2 import KilledUnitsReward
from urnai.agents.sc2_agent import SC2Agent
from urnai.agents.states.sc2 import TVTUnitStackingState
from urnai.envs.sc2 import SC2Env
from urnai.models.algorithms.ddql import DoubleDeepQLearning
from urnai.models.model_builder import ModelBuilder
from urnai.trainers.trainer import Trainer

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

"""
Change 'SC2PATH' to your local SC2 installation path.
This only needs to be done once on each machine!
If you used the default installation path, you may ignore this step.
For more information see https://github.com/deepmind/pysc2#get-starcraft-ii
"""
# os.environ['SC2PATH'] = 'D:/Program Files (x86)/StarCraft II'


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def declare_trainer():
    env = SC2Env(map_name='Simple64', render=False, step_mul=16, player_race='terran',
                 enemy_race='terran', difficulty='very_easy')

    action_wrapper = SimpleTerranWrapper(use_atk_grid=True, atk_grid_x=4, atk_grid_y=4)
    state_builder = TVTUnitStackingState()

    helper = ModelBuilder()
    helper.add_input_layer(nodes=82)
    helper.add_output_layer()
    print(helper.get_model_layout())

    dq_network = DoubleDeepQLearning(action_wrapper=action_wrapper, state_builder=state_builder,
                                     build_model=helper.get_model_layout(), use_memory=True,
                                     gamma=0.99, learning_rate=0.001, memory_maxlen=100000,
                                     min_memory_size=64, lib='keras',
                                     epsilon_decay=0.99999, epsilon_start=1.0, epsilon_min=0.005,
                                     epsilon_linear_decay=False, per_episode_epsilon_decay=False)

    agent = SC2Agent(dq_network, KilledUnitsReward())

    trainer = Trainer(env, agent, save_path='urnai/models/saved',
                      file_name='terran_ddql_newpredict_customfit',
                      save_every=12, enable_save=True, relative_path=True, reset_epsilon=False,
                      max_training_episodes=4, max_steps_training=800,
                      max_test_episodes=1, max_steps_testing=100, rolling_avg_window_size=3)
    return trainer


def main(unused_argv):
    try:
        trainer = declare_trainer()
        trainer.train()
        # trainer.play()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
