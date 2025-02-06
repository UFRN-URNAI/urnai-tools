import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import optuna
from absl import app
from optuna.integration.wandb import WeightsAndBiasesCallback

from experiments.solves.solve_collectables_sb3 import declare_trainer


def objective(trial):

    config_dict = {
        "policy":"MlpPolicy",
        "model_save_name": "PPOMlp"}
    
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    n_steps = trial.suggest_int("n_steps", 128, 2048, step=128)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, step=0.0001)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0, step=0.01)
    
    hyperparameters = dict(
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda
    )

    trainer = declare_trainer(config_dict, hyperparameters)
    trainer.train_model(timesteps = 1000, repeat_times = 1)
    mean_reward, _ = trainer.test_model(episodes = 5, return_episode_rewards = False,
                                         wandb_log = False)
    trainer.train_env.close()

    return mean_reward

def print_study_results(study):
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
