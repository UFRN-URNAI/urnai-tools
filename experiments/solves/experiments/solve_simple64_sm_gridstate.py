import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

from absl import app

from urnai.envs.sc2 import SC2Env
from urnai.trainers.trainer import Trainer
from urnai.agents.sc2_agent import SC2Agent
from urnai.agents.actions.sc2_wrapper import SimpleTerranWrapper, SimpleMarineWrapper
from urnai.agents.rewards.sc2 import KilledUnitsReward, KilledUnitsRewardImproved, TStarBotReward
from urnai.agents.states.sc2 import Simple64GridState
from urnai.models.model_builder import ModelBuilder
from urnai.models.algorithms.ddql import DoubleDeepQLearning

def declare_trainer():
    state_builder = Simple64GridState(grid_size=4)
    action_wrapper = SimpleMarineWrapper()

    env = SC2Env(map_name="Simple64", render=False, self_play=False, 
                step_mul=16, realtime=False, player_race="terran", 
                enemy_race="random", difficulty="very_easy")

    helper = ModelBuilder()
    helper.add_input_layer(100)
    helper.add_fullyconn_layer(50)
    helper.add_output_layer()

    dq_network = DoubleDeepQLearning(lib="pytorch", action_wrapper=action_wrapper,
                                    state_builder=state_builder, build_model=helper.get_model_layout(),
                                    epsilon_start=0.8, epsilon_decay=0.999, epsilon_linear_decay=True,
                                    per_episode_epsilon_decay=True, use_memory=True,
                                    epsilon_min=0.05, epsilon_decay_ep_start=200,
                                    learning_rate=0.00013, min_memory_size=50000, memory_maxlen=120000)

    agent = SC2Agent(dq_network, KilledUnitsRewardImproved())

    trainer = Trainer(env, agent, save_path='urnai/models/saved/sm_killed_reward_improved/',
                    file_name="sm_killed_reward_vs_random_lr_04",
                    save_every=100, enable_save=False, relative_path=True,
                    max_training_episodes=5000, max_steps_training=1200,
                    max_test_episodes=200, max_steps_testing=1200)

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
