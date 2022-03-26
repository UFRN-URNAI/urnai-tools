import pathlib
import sys

from absl import app
from urnai.agents.generic_agent import GenericAgent
from urnai.agents.rewards.gym import FrozenlakeJiexunseeReward
from urnai.agents.states.gym import FrozenLakeState
from urnai.envs.gym import GymEnv
from urnai.models.ddqn_keras import DDQNKeras
from urnai.models.model_builder import ModelBuilder
from urnai.trainers.trainer import Trainer

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))


def declare_trainer():
    env = GymEnv(id='FrozenLake-v0')

    action_wrapper = env.get_action_wrapper()
    state_builder = FrozenLakeState()

    helper = ModelBuilder()
    helper.add_input_layer(nodes=12)
    helper.add_output_layer()

    dq_network = DDQNKeras(action_wrapper=action_wrapper, state_builder=state_builder,
                           learning_rate=0.005, gamma=0.90, use_memory=False,
                           per_episode_epsilon_decay=True, build_model=helper.get_model_layout())

    agent = GenericAgent(dq_network, FrozenlakeJiexunseeReward())

    trainer = Trainer(env, agent, save_path='urnai/models/saved',
                      file_name='frozenlake_ddql_oldpredict',
                      save_every=50, enable_save=True, relative_path=True, reset_epsilon=False,
                      max_training_episodes=350, max_steps_training=10000,
                      max_test_episodes=50, max_steps_testing=10000, rolling_avg_window_size=25)

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
