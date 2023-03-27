import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from absl import app
from urnai.agents.actions.sc2_wrapper import SimpleTerranWrapper
from urnai.agents.rewards.sc2 import KilledUnitsReward
from urnai.agents.sc2_agent import SC2Agent
from urnai.agents.states.sc2 import Simple64GridState
from urnai.envs.sc2 import SC2Env
from urnai.models.ddqn_keras import DDQNKeras
from urnai.models.model_builder import ModelBuilder
from urnai.trainers.trainer import Trainer
from urnai.models.algorithms.ddql import DoubleDeepQLearning



def declare_trainer():
    env = SC2Env(map_name='Simple64', render=False,
                 step_mul=16, player_race='terran',
                 enemy_race='random', difficulty='very_easy', self_play=True)
	
	
    action_wrapper = SimpleTerranWrapper()
    state_builder = Simple64GridState(grid_size=4)

    helper = ModelBuilder()
    helper.add_input_layer(nodes=50)
    helper.add_fullyconn_layer(50)
    helper.add_output_layer()

    dq_network = DoubleDeepQLearning(action_wrapper=action_wrapper,
                                    state_builder=state_builder,
                                    build_model=helper.get_model_layout(),
                                    per_episode_epsilon_decay=False,
                                    gamma=0.99, learning_rate=0.0001,
                                    epsilon_decay=0.99999, epsilon_min=0.005,
                                    memory_maxlen=100000, min_memory_size=2000,
                                    lib="keras")

    agent = SC2Agent(dq_network, KilledUnitsReward())

    trainer = Trainer(env, agent, save_path='urnai/models/saved',
                      file_name='terran_selfplay_v3_lr03',
                      save_every=1, enable_save=True, relative_path=True,
                      max_training_episodes=3000, max_steps_training=1200,
                      max_test_episodes=150, max_steps_testing=1200)
    return trainer


def main(unused_argv):
    try:
        trainer = declare_trainer()
        #trainer.train()
        trainer.play()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)



##### COMANDO PARA RODAR ESSE CÓDIGO:
#  python c:/Users/thuan/urnai-tools-master/experiments/solves/solve_simple64_veryeasy.py
