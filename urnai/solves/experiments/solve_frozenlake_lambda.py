import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

from absl import app
from urnai.envs.gym import GymEnv
from urnai.trainers.trainer import Trainer
from urnai.trainers.trainer import TestParams
from urnai.agents.generic_agent import GenericAgent
from urnai.agents.actions.gym_wrapper import GymWrapper
from urnai.agents.rewards.gym import FrozenlakeReward, FrozenlakeJiexunseeReward
from urnai.agents.states.gym import FrozenLakeState
from urnai.models.ddqn_keras import DDQNKeras
from urnai.models.algorithms.dql import DeepQLearning
from urnai.models.algorithms.dql_lambda import DeepQLearningLambda
from urnai.models.model_builder import ModelBuilder
from datetime import datetime

def declare_trainer():
    env = GymEnv(id="FrozenLake-v0")

    action_wrapper = env.get_action_wrapper() 
    state_builder = FrozenLakeState()

    training_date = str(datetime.now()).replace(" ","_").replace(":","_").replace(".","_")

    helper = ModelBuilder()
    helper.add_input_layer(nodes=25)
    helper.add_output_layer()
    
    #dq_network = DDQNKeras(action_wrapper=action_wrapper, state_builder=state_builder, learning_rate=0.005, gamma=0.90, use_memory=False, per_episode_epsilon_decay = True, build_model=helper.get_model_layout())
    dq_network = DeepQLearningLambda(action_wrapper=action_wrapper, state_builder=state_builder, build_model=helper.get_model_layout(),
                        lamb=0.9, gamma=0.99, learning_rate=0.005, memory_maxlen=10000, min_memory_size=64, lib="keras_e_traces",
                        epsilon_decay=0.999, epsilon_start=1.0, epsilon_min=0.005, epsilon_linear_decay=False, per_episode_epsilon_decay=False)

    # dq_network = DeepQLearning(action_wrapper=action_wrapper, state_builder=state_builder, build_model=helper.get_model_layout(), 
    #                 gamma=0.99, learning_rate=0.005, memory_maxlen=10000, min_memory_size=64, lib="keras_e_traces",
    #                 epsilon_decay=0.999, epsilon_start=1.0, epsilon_min=0.005, epsilon_linear_decay=False, per_episode_epsilon_decay=False)

    agent = GenericAgent(dq_network, FrozenlakeJiexunseeReward())

    # trainer = Trainer(env, agent, file_name=training_date + os.path.sep + "frozenlake_test_ddqnKeras", save_every=1000, enable_save=True,
    #                     max_training_episodes = 1000, max_steps_training=500,
    #                     max_test_episodes=10, max_steps_testing=500)

    trainer = Trainer(env, agent, save_path='urnai/models/saved', file_name="frozenlake_dql_lambda4_batch1",
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
