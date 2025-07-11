import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from absl import app
from gymnasium import spaces
from pysc2.env import sc2_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import wandb
from urnai.environments.stablebaselines3.custom_env_collectables import (
    CustomEnvCollectables,
)
from urnai.loggers.wandb_logger import WandbLogger
from urnai.sc2.actions.collectables import CollectablesActionSpace
from urnai.sc2.environments.sc2environment import SC2Env
from urnai.sc2.rewards.collectables import CollectablesReward
from urnai.sc2.states.collectables import CollectablesState
from urnai.trainers.stablebaselines3_trainer import SB3Trainer

EPISODE_MINUTES = 2
STEPS_PER_SECOND = 16
STEPS_PER_MINUTE = int(STEPS_PER_SECOND * 60)
MAX_STEPS = EPISODE_MINUTES * STEPS_PER_MINUTE


def declare_wandb_run(config_dict : dict, run_id : str = None):

    wandb_run = wandb.init(
        project='solve_collectables_sb3',
        config=config_dict,
        name=config_dict['model_save_name'],
        sync_tensorboard=True,
        resume="must" if run_id else None,
        id=run_id
    )

    return wandb_run
    
def declare_trainer(config_dict: dict, hyperparameters: dict = None):
    players = [sc2_env.Agent(sc2_env.Race.terran)]
    action_space = spaces.Discrete(n=4, start=0)
    observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    step_mult = 12
    use_invalid_action_masking = config_dict.get('invalid_action_masking', True)

    logger = WandbLogger()  # Uma única instância de logger compartilhada

    # SC2Env separados para treino e avaliação
    train_sc2_env = SC2Env(map_name='CollectMineralShards', step_mul=step_mult, 
                           players=players, visualize=False)
    eval_sc2_env = SC2Env(map_name='CollectMineralShards', step_mul=step_mult, 
                          players=players, visualize=False)

    # Instâncias separadas dos componentes com estado
    train_state = CollectablesState()
    train_action_space = CollectablesActionSpace()
    train_reward = CollectablesReward()

    eval_state = CollectablesState()
    eval_action_space = CollectablesActionSpace()
    eval_reward = CollectablesReward()

    # CustomEnv separados para treino e avaliação
    train_custom_env = CustomEnvCollectables(train_sc2_env, train_state, 
                                             train_action_space, train_reward,
                                             observation_space, action_space, logger,
                                             step_mult, MAX_STEPS)
    eval_custom_env = CustomEnvCollectables(eval_sc2_env, eval_state, 
                                            eval_action_space, eval_reward, 
                                            observation_space, action_space, logger,
                                            step_mult, MAX_STEPS)

    # Wrappers Monitor
    train_env = Monitor(train_custom_env)
    eval_env = Monitor(eval_custom_env)

    models_dir = f"/home/mambauser/saves/models/{config_dict['model_save_name']}"
    logdir = "/home/mambauser/saves/logs"

    model = PPO(config_dict['policy'], train_env, verbose=1,
                tensorboard_log=logdir,
                **(hyperparameters if hyperparameters is not None else {}))
    
    if use_invalid_action_masking:
        def mask_fn(env: CustomEnvCollectables) -> np.ndarray:
            return env.unwrapped.get_action_mask()

        train_env = ActionMasker(train_env, mask_fn)
        eval_env = ActionMasker(eval_env, mask_fn)

        model = MaskablePPO(config_dict['policy'],train_env,verbose=1,
            tensorboard_log=logdir,
            **(hyperparameters if hyperparameters is not None else {})
        )

    trainer = SB3Trainer(
        train_env, eval_env, models_dir, logdir, model,
        config_dict['model_save_name'], logger=logger, 
        use_masking=use_invalid_action_masking
    )

    return trainer

def main(unused_argv):
    try:
        config_dict = {
            "policy":"MlpPolicy",
            "model_save_name": "MaskablePPO",
            "invalid_action_masking": True}
        wandb_run = declare_wandb_run(config_dict)
        trainer = declare_trainer(config_dict)
        # trainer.load_most_recent_model(trainer.models_dir)
        trainer.alternate_train_test(
            iterations=10000000,
            train_steps=50*(MAX_STEPS/ (12)), 
            test_episodes=10,
            callback=None,
            return_episode_rewards=True, wandb_log=True
        )
        wandb_run.finish()
    except KeyboardInterrupt:
        print("Training interrupted by user")

if __name__ == '__main__':
    app.run(main)