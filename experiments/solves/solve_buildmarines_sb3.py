import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from copy import deepcopy
from absl import app
from gymnasium import spaces
from pysc2.env import sc2_env
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from typing import Callable

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

from experiments.solves.experiment_progress_recorder import recorder

EPISODE_MINUTES = 1
STEPS_PER_SECOND = 22  # padrão do StarCraft II (aproximado)
STEPS_PER_MINUTE = int(STEPS_PER_SECOND * 60)
MAX_STEPS = EPISODE_MINUTES * STEPS_PER_MINUTE
ITERATIONS = 2
WANDB_ENABLED = False


def exponential_schedule(initial_value : float, final_value : float) -> Callable[[float], float]:
    def func(unused_progress_remaining : float) -> float:
        base = final_value + (1 - initial_value)

        global_progres_remaining = recorder.get_progress()
        new_value = base ** global_progres_remaining - (1 - initial_value)

        print("\nSCHEDULE UPDATING VALUE TO: ", new_value,
            " | ", global_progres_remaining * 100, "% TO FINAL VALUE\n")
        
        return new_value
    return func

def declare_wandb_run(config_dict : dict, run_id : str = None):

    wandb_run = wandb.init(
        project='solve_buildmarines_debug',
        config=config_dict,
        name=config_dict['model_save_name'],
        sync_tensorboard=True,
        resume="must" if run_id else None,
        id=run_id
    )

    return wandb_run

def make_history_space_of(space : spaces.Space, history_size : int):
    dict_ = {}
    for i in range(history_size):
        dict_[str(i)] = deepcopy(space)
    return spaces.Dict(dict_)

def declare_trainer(config_dict: dict, hyperparameters: dict = None):
    players = [sc2_env.Agent(sc2_env.Race.terran)]
    action_space = spaces.Discrete(n=4, start=0)
    observation_space = make_history_space_of(
        spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=float),
          history_size = config_dict['history_size'])
    step_mult = 32
    use_invalid_action_masking = config_dict.get('invalid_action_masking', True)

    logger = None
    if WANDB_ENABLED:
        logger = WandbLogger()  # Uma única instância de logger compartilhada

    # SC2Env separados para treino e avaliação
    train_sc2_env = SC2Env(map_name='BuildMarines', step_mul=step_mult, players=players)
    eval_sc2_env = SC2Env(map_name='BuildMarines', step_mul=step_mult, players=players)

    # Instâncias separadas dos componentes com estado
    train_state = BuildMarinesState(history_length=len(observation_space.spaces))
    train_action_space = BuildMarinesActionSpace()
    train_reward = BuildMarinesReward(config_dict)

    eval_state = BuildMarinesState(history_length=len(observation_space.spaces))
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

    models_dir = f"/home/mambauser/urnai/saves/models/{config_dict['model_save_name']}"
    logdir = "/home/mambauser/urnai/saves/logs"

    model = PPO(config_dict['policy'], train_env, verbose=0,
                clip_range=exponential_schedule(initial_value=0.8, final_value=0.2),
                tensorboard_log=logdir,
                **(hyperparameters if hyperparameters is not None else {}))
    
    if use_invalid_action_masking:
        def mask_fn(env: CustomEnvBuildMarines) -> np.ndarray:
            return env.unwrapped.get_action_mask()

        train_env = ActionMasker(train_env, mask_fn)
        eval_env = ActionMasker(eval_env, mask_fn)

        model = MaskablePPO(config_dict['policy'],train_env,verbose=0,
            clip_range=exponential_schedule(initial_value=0.8, final_value=0.2),
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
            "policy":"MultiInputPolicy",
            "model_save_name": "test-performance-singleton",
            "w_supply": 0, #0.1
            "w_barrack": 0, #15/80
            "w_marine": 10,
            "penalty_no_supply": 0,
            "penalty_no_barrack": 0,
            "invalid_action_masking": True,
            "history_size" : 1
        }
        if WANDB_ENABLED:
            wandb_run = declare_wandb_run(config_dict)
        
        trainer = declare_trainer(config_dict)
        #trainer.load_most_recent_model(trainer.models_dir)
        trainer.alternate_train_test(
            starting_iteration=0,
            iterations=ITERATIONS,
            train_steps=int(1 * MAX_STEPS / 32),
            test_episodes=1,
            wandb_log=WANDB_ENABLED
        )

        if WANDB_ENABLED:
            wandb_run.finish()
        
    except KeyboardInterrupt:
        print("Training interrupted by user")

if __name__ == '__main__':
    app.run(main)