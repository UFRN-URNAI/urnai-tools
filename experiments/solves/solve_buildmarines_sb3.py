import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from absl import app
from gymnasium import spaces
from pysc2.env import sc2_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import wandb
from urnai.environments.stablebaselines3.custom_env_buildmarines import (
    CustomEnvBuildMarines,
)
from urnai.loggers.wandb_logger import WandbLogger
from urnai.sc2.actions.buildmarines import BuildMarinesActionSpace
from urnai.sc2.environments.sc2environment import SC2Env
from urnai.sc2.rewards.buildmarines import BuildMarinesReward
from urnai.sc2.states.buildmarines import BuildMarinesState
from urnai.trainers.stablebaselines3_trainer import SB3Trainer

EPISODE_MINUTES = 15
STEPS_PER_SECOND = 22  # padrão do StarCraft II (aproximado)
STEPS_PER_MINUTE = int(STEPS_PER_SECOND * 60)
MAX_STEPS = EPISODE_MINUTES * STEPS_PER_MINUTE


def declare_wandb_run(config_dict : dict, run_id : str = None):

    wandb_run = wandb.init(
        project='solve_buildmarines',
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
    observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=float)
    step_mult = 32

    logger = WandbLogger()  # Uma única instância de logger compartilhada

    # SC2Env separados para treino e avaliação
    train_sc2_env = SC2Env(map_name='BuildMarines', step_mul=step_mult, players=players)
    eval_sc2_env = SC2Env(map_name='BuildMarines', step_mul=step_mult, players=players)

    # Instâncias separadas dos componentes com estado
    train_state = BuildMarinesState()
    train_action_space = BuildMarinesActionSpace()
    train_reward = BuildMarinesReward(config_dict)

    eval_state = BuildMarinesState()
    eval_action_space = BuildMarinesActionSpace()
    eval_reward = BuildMarinesReward(config_dict)

    # CustomEnv separados para treino e avaliação
    train_custom_env = CustomEnvBuildMarines(train_sc2_env, train_state, 
                                             train_action_space, train_reward,
                                             observation_space, action_space, logger,
                                             step_mult, MAX_STEPS)
    eval_custom_env = CustomEnvBuildMarines(eval_sc2_env, eval_state, 
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

    trainer = SB3Trainer(
        train_env, eval_env, models_dir, logdir, model,
        config_dict['model_save_name'], logger=logger
    )

    return trainer

def main(unused_argv):
    try:
        config_dict = {
            "policy":"MlpPolicy",
            "model_save_name": "TestMoreBarracks2",
            "w_supply": 1.0,
            "w_barrack": 30.0,
            "w_marine": 1.5,
            "penalty_no_supply": 0.0,
            "penalty_no_barrack": 0.0
        }
        wandb_run = declare_wandb_run(config_dict)
        trainer = declare_trainer(config_dict)
        # trainer.load_most_recent_model(trainer.models_dir)
        trainer.alternate_train_test(
            iterations=100000,
            train_steps= int(50 * MAX_STEPS / 32),
            test_episodes=10,
            callback=None,
            return_episode_rewards=True, wandb_log=True
        )
        # trainer.test_model(
        #     episodes=100, deterministic=True, render=False,
        #     wandb_log=False
        # )
        # trainer.train_model(
        #     timesteps=100000, log_interval=1,
        #     reset_num_timesteps=False, progress_bar=True, 
        #     repeat_times=1, start_from=1, callback=WandbCallback()
        # )
        wandb_run.finish()
    except KeyboardInterrupt:
        print("Training interrupted by user")

if __name__ == '__main__':
    app.run(main)