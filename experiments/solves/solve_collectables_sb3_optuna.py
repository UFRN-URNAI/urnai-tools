import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import optuna
from absl import app
from gymnasium import spaces
from optuna.integration.wandb import WeightsAndBiasesCallback
from pysc2.env import sc2_env
from stable_baselines3 import PPO

from urnai.environments.stablebaselines3.custom_env import CustomEnv
from urnai.sc2.actions.collectables import CollectablesActionSpace
from urnai.sc2.environments.sc2environment import SC2Env
from urnai.sc2.rewards.collectables import CollectablesReward
from urnai.sc2.states.collectables import CollectablesMethod, CollectablesState
from urnai.trainers.stablebaselines3_trainer import SB3Trainer


def declare_trainer(trial, config_dict : dict):
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

    models_dir = f"saves/models/{config_dict['model_save_name']}"
    logdir = "saves/logs"

    # Hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    n_steps = trial.suggest_int("n_steps", 128, 2048, step=128)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, step=0.0001)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0, step=0.01)

    model=PPO(config_dict['policy'], custom_env, verbose=1, tensorboard_log=logdir,
                learning_rate=learning_rate,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda)

    trainer = SB3Trainer(
        custom_env, models_dir, logdir, model, config_dict['model_save_name']
    )

    return trainer

def objective(trial):

    config_dict = {
        "policy":"MlpPolicy",
        "model_save_name": "PPOMlp"}
    trainer = declare_trainer(trial, config_dict)

    # trainer.load_model(f"{trainer.models_dir}/100000")
    mean_reward, _ = trainer.train_model(timesteps = 1000,
                                         repeat_times = 1)

    return mean_reward

def print_study_results(study):
    # Print best hyperparameters
        print(f"Best hyperparameters:{study.best_params}")

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

def main(unused_argv):
    try:

        run_id = None
        wandb_kwargs = {"project": "solve_collectables",
                        "resume" : "must" if run_id else None,
                        "id" : run_id}
        wandbc = WeightsAndBiasesCallback(metric_name="mean_reward",
                                           wandb_kwargs=wandb_kwargs)

        study_name = "cool_study"
        study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                storage=f"sqlite:///{study_name}.db",
                load_if_exists=True)
        
        study.optimize(objective, n_trials=5, callbacks=[wandbc])

        print_study_results(study)

    except KeyboardInterrupt:
        print("Training interrupted by user")

if __name__ == '__main__':
    app.run(main)
