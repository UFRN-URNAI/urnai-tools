import os

from stable_baselines3.common.base_class import BaseAlgorithm
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.type_aliases import MaybeCallback

from urnai.environments.stablebaselines3.custom_env import CustomEnv
from urnai.loggers.logger_base import LoggerBase

from experiments.solves.experiment_progress_recorder import recorder

class SB3Trainer:
    def __init__(self, train_env : CustomEnv, eval_env : CustomEnv, models_dir : str, 
                 logdir : str, model : BaseAlgorithm, model_name : str, 
                 logger : LoggerBase = None, use_masking: bool = True):
        self.train_env = train_env
        self.eval_env = eval_env
        self.models_dir = models_dir
        self.model = model
        self.model_name = model_name
        self.logger = logger
        self.use_masking = use_masking

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if not os.path.exists(logdir):
            os.makedirs(logdir)
    
    def load_model(self, model_path):
        self.model = self.model.load(model_path, env = self.train_env)

    def load_most_recent_model(self, model_path):
        save_files = list(filter(lambda filename : ".save" in  filename,
                                  os.listdir(model_path)))
        
        if len(save_files) == 0:
            raise Exception(f"No models found in {model_path}")
        else:
            def only_digits(filename):
                return int(''.join(c for c in filename if c.isdigit()))
            save_files.sort(reverse=True, key=only_digits)
            self.load_model(f"{model_path}/{save_files[0]}")
    
    def train_model(
            self, timesteps: int = 10000, log_interval: int = 1,
            reset_num_timesteps: bool = False, progress_bar: bool = False, 
            repeat_times:int = 1, start_from:int = 1, callback : MaybeCallback = None
        ) -> None:

        if self.logger:
            self.logger.set_mode(train = True)
        
        for repeat_time in range(repeat_times):
            self.model.learn(total_timesteps = timesteps, callback = callback,
                            log_interval = log_interval,
                            reset_num_timesteps = reset_num_timesteps,
                            progress_bar = progress_bar,
                            tb_log_name = self.model_name)
            time_id = timesteps*(repeat_time + start_from)
            self.model.save(f"{self.models_dir}/{time_id}.save")
    
    def test_model(
            self, episodes : int = 10, deterministic: bool = True,
            render = False, callback = None, reward_threshold = None,
            return_episode_rewards = False, warn = True, wandb_log = False
        ) -> tuple[float, float] | tuple[list[float], list[int]]:

        if self.logger:
            self.logger.set_mode(train = False)

        episode_rewards = evaluate_policy(model = self.model, env = self.eval_env, 
                        n_eval_episodes=episodes, 
                        deterministic=deterministic, 
                        render=render,
                        callback=callback,
                        reward_threshold=reward_threshold,
                        return_episode_rewards=return_episode_rewards,
                        warn=warn,
                        use_masking=self.use_masking
                        )
        
        return episode_rewards

    def alternate_train_test(
            self, starting_iteration : int = 0,
            iterations : int = 100, train_steps : int = 10000, 
            train_repeat_times : int = 1, test_episodes : int = 100, 
            callback : MaybeCallback = None, return_episode_rewards : bool = True,
            wandb_log : bool = True
        ) -> None:
        for iteration in range(starting_iteration, iterations):
            recorder.set_progress(iteration/iterations)

            print(f"Iteration {iteration+1}/{iterations}")
            print(f"Training for {train_steps} steps")
            self.train_model(
                timesteps=train_steps, repeat_times=train_repeat_times,
                start_from=iteration*train_repeat_times +1, callback=callback
            )
            print(f"Testing for {test_episodes} episodes")
            self.test_model(episodes = test_episodes,
                            return_episode_rewards = return_episode_rewards,
                            wandb_log = wandb_log)

    def close(self) -> None:
        self.train_env.close()
        self.eval_env.close()
    