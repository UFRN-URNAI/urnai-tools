import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

from absl import app
from urnai.envs.sc2 import SC2Env
from urnai.trainers.trainer import Trainer
from urnai.agents.sc2_agent import SC2Agent
from urnai.agents.rewards.default import PureReward
from urnai.agents.rewards.sc2 import MoveToBeaconProximity, MoveToBeaconDirection
from urnai.agents.states.sc2 import DefeatRoachesState
from urnai.agents.actions.mo_spatial_terran_wrapper import (
    DefeatRoachesSimplified, DefeatRoachesSimplified_noop, 
    DefeatRoachesFull, DefeatRoachesActionGroup
)
from urnai.models.ddqn_keras_mo import DDQNKerasMO
from urnai.models.model_builder import ModelBuilder

from urnai.utils.reporter import Reporter as rp

""" Change "sc2_local_path" to your local SC2 installation path. 
If you used the default installation path, you may ignore this step.
For more information consult https://github.com/deepmind/pysc2#get-starcraft-ii 
"""
# sc2_local_path = "D:/Program Files (x86)/StarCraft II"

def declare_trainer(): 
    env = SC2Env(map_name="DefeatRoaches", realtime=False, render=False, step_mul=8, player_race="terran", enemy_race="terran", difficulty="very_easy")

    #action_wrapper = DefeatRoachesActionGroup(4, 10, 10, random_uniform=True)
    action_wrapper = DefeatRoachesSimplified_noop(10, 10, random_uniform=True)
    state_builder = DefeatRoachesState()

    helper = ModelBuilder()
    helper.add_input_layer(nodes=38)
    helper.add_fullyconn_layer(nodes=22)
    #helper.add_fullyconn_layer(nodes=28)
    helper.add_output_layer()

    print(helper.get_model_layout())

    dq_network = DDQNKerasMO(action_wrapper=action_wrapper, state_builder=state_builder, build_model=helper.get_model_layout(), per_episode_epsilon_decay=True,
                        learning_rate=0.001, epsilon_decay=0.995, epsilon_min=0.005, memory_maxlen=50000, min_memory_size=128, batch_size=32)
    
    # Terran agent
    agent = SC2Agent(dq_network, PureReward())

    trainer = Trainer(env, agent, save_path='../agentes-treinados-tcc/defeat-roaches', file_name="defeatroaches_simple10x10act_fixed",
                    save_every=100, enable_save=True, relative_path=True, reset_epsilon=False,
                    max_training_episodes=600, max_steps_training=300,
                    max_test_episodes=10, max_steps_testing=300, rolling_avg_window_size=20)
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
