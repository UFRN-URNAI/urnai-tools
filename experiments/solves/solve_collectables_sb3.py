import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from absl import app
from gymnasium import spaces
from pysc2.env import sc2_env
from stable_baselines3 import PPO

from urnai.environments.stablebaselines3.custom_env import CustomEnv
from urnai.sc2.actions.collectables import CollectablesActionSpace
from urnai.sc2.environments.sc2environment import SC2Env
from urnai.sc2.rewards.collectables import CollectablesReward
from urnai.sc2.states.collectables import CollectablesMethod, CollectablesState
from urnai.trainers.stablebaselines3_trainer import SB3Trainer


def declare_trainer():
    players = [sc2_env.Agent(sc2_env.Race.terran)]
    env = SC2Env(map_name='CollectMineralShards', visualize=False, 
                step_mul=16, players=players)
    state = CollectablesState(method=CollectablesMethod.STATE_MAP)
    urnai_action_space = CollectablesActionSpace()
    reward = CollectablesReward()

    # Define action and observation space
    action_space = spaces.Discrete(n=4, start=0)
    observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

    # Create the custom environment
    custom_env = CustomEnv(env, state, urnai_action_space, reward, observation_space, 
                        action_space)

    model_name = "PPOMlp"
    models_dir = f"saves/models/{model_name}"
    logdir = "saves/logs"

    conf_dict = {
             "policy":"MlpPolicy",
             "model_save_name": model_name}

    model=PPO("CnnPolicy", custom_env, verbose=1, tensorboard_log=logdir)

    trainer = SB3Trainer(custom_env, models_dir, logdir, model, model_name, 
                         "solve_collectables", conf_dict)

    return trainer

def main(unused_argv):
    try:
        trainer = declare_trainer()
        # trainer.load_model(f"{trainer.models_dir}/100000")
        trainer.alternate_train_test(iterations=100, train_steps=10000, test_steps=1000)
    except KeyboardInterrupt:
        print("Training interrupted by user")

if __name__ == '__main__':
    app.run(main)