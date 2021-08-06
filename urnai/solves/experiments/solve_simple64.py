import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

from absl import app
from pysc2.env import sc2_env
from urnai.envs.sc2 import SC2Env
from urnai.trainers.trainer import Trainer
from urnai.agents.sc2_agent import SC2Agent
from urnai.agents.actions.sc2_wrapper import SimpleTerranWrapper
from urnai.agents.actions.mo_spatial_terran_wrapper import MOspatialTerranWrapper
from urnai.agents.rewards.sc2 import KilledUnitsReward
from urnai.agents.states.sc2 import Simple64GridState, SimpleCroppedGridState, TVTUnitStackingState, MultipleUnitGridState, TVTUnitStackingEnemyGridState
from urnai.models.ddqn_keras import DDQNKeras
from urnai.models.ddqn_keras_mo import DDQNKerasMO
from urnai.models.algorithms.dql import DeepQLearning
from urnai.models.algorithms.ddql import DoubleDeepQLearning
from urnai.models.memory_representations.neural_network.keras import DNNCustomModelOverrideExample
from urnai.utils.functions import query_yes_no
from urnai.models.model_builder import ModelBuilder

from urnai.utils.reporter import Reporter as rp

"""
Change "SC2PATH" to your local SC2 installation path.
This only needs to be done once on each machine!
If you used the default installation path, you may ignore this step.
For more information see https://github.com/deepmind/pysc2#get-starcraft-ii 
"""
# os.environ["SC2PATH"] = "D:/Program Files (x86)/StarCraft II"

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def declare_trainer():
    env = SC2Env(map_name="Simple64", render=False, step_mul=16, player_race="terran", enemy_race="terran", difficulty="very_easy")
    
    action_wrapper = SimpleTerranWrapper(use_atk_grid=False, atk_grid_x=4, atk_grid_y=4)
    state_builder = TVTUnitStackingState()
    
    helper = ModelBuilder()
    helper.add_input_layer(nodes=60)
    helper.add_fullyconn_layer(nodes=75)
    helper.add_output_layer()
    print(helper.get_model_layout())
    
    dq_network = DoubleDeepQLearning(action_wrapper=action_wrapper, state_builder=state_builder, build_model=helper.get_model_layout(), use_memory=True,
                        gamma=0.99, learning_rate=0.001, memory_maxlen=100000, min_memory_size=64, lib="keras",
                        epsilon_decay=0.9994, epsilon_start=0.6, epsilon_min=0.005, epsilon_linear_decay=True, per_episode_epsilon_decay=True)
    
    agent = SC2Agent(dq_network, KilledUnitsReward())

    # trainer = Trainer(env, agent, save_path='/home/lpdcalves/', file_name="tvt_veasy_newactwrapper_t1",
    #                 save_every=200, enable_save=True, relative_path=False, reset_epsilon=False,
    #                 max_training_episodes=3000, max_steps_training=1500,
    #                 max_test_episodes=100, max_steps_testing=1500, rolling_avg_window_size=50)

    trainer = Trainer(env, agent, save_path='urnai/models/saved', file_name="terran_ddql_tvtunitstacking1",
                    save_every=6, enable_save=True, relative_path=True, reset_epsilon=False,
                    max_training_episodes=1, max_steps_training=1200,
                    max_test_episodes=1, max_steps_testing=100, rolling_avg_window_size=50)
    return trainer

def main(unused_argv):
    try:
        trainer = declare_trainer()
        trainer.train()
        #trainer.play()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
