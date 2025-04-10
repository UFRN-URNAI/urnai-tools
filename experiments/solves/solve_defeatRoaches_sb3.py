import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from absl import app
from gymnasium import spaces
from pysc2.env import sc2_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import wandb
from urnai.environments.stablebaselines3.custom_env import CustomEnv
from urnai.sc2.actions.defeatRoaches import DefeatRoachesActionSpace
from urnai.sc2.environments.sc2environment import SC2Env
from urnai.sc2.rewards.defeatRoaches import DefeatRoachesReward
from urnai.sc2.states.defeatRoaches import DefeatRoachesState
from urnai.trainers.stablebaselines3_trainer import SB3Trainer
from wandb.integration.sb3 import WandbCallback


def declare_wandb_run(config_dict : dict, run_id : str = None):

    wandb_run = wandb.init(
        project='solve_defeatRoaches',
        config=config_dict,
        name=config_dict['model_save_name'],
        sync_tensorboard=True,
        resume="must" if run_id else None,
        id=run_id
    )

    return wandb_run
    
def declare_trainer(config_dict : dict, hyperparameters : dict = None):
    players = [sc2_env.Agent(sc2_env.Race.terran)]
    action_space = spaces.Discrete(n = 12, start = 0)
    observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

    env = SC2Env(map_name='DefeatRoaches', visualize=False, 
                step_mul=16, players=players)
    state = DefeatRoachesState()
    urnai_action_space = DefeatRoachesActionSpace()
    reward = DefeatRoachesReward()

    custom_env = CustomEnv(env, state, urnai_action_space, reward,
                            observation_space, action_space)
    train_env = Monitor(custom_env)
    eval_env = Monitor(custom_env)

    models_dir = f"saves/models/{config_dict['model_save_name']}"
    logdir = "saves/logs"

    model=PPO(config_dict['policy'], custom_env, verbose=1,
        tensorboard_log=logdir,
        **(hyperparameters if hyperparameters is not None else {}))

    trainer = SB3Trainer(
        train_env, eval_env, models_dir, logdir, model, config_dict['model_save_name']
    )

    return trainer

def main(unused_argv):
    try:
        config_dict = {
            "policy":"MlpPolicy",
            "model_save_name": "PPOMlp_DefeatRoaches_teste"}
        #wandb_run = declare_wandb_run(config_dict)
        trainer = declare_trainer(config_dict)
        #trainer.load_most_recent_model(trainer.models_dir)
        trainer.alternate_train_test(
            iterations=100, train_steps=10000, test_episodes=20,
            callback=None, #TODO: put wandb back
            return_episode_rewards=True, wandb_log=False
        )
        #wandb_run.finish()
    except KeyboardInterrupt:
        print("Training interrupted by user")

if __name__ == '__main__':
    app.run(main)