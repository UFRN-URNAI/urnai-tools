import os

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.type_aliases import MaybeCallback

import wandb
from urnai.environments.stablebaselines3.custom_env import CustomEnv


class SB3Trainer:
    def __init__(self, custom_env : CustomEnv, models_dir : str, logdir : str, 
                 model : BaseAlgorithm, model_name : str):
        self.custom_env = custom_env
        self.models_dir = models_dir
        self.model = model
        self.model_name = model_name

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if not os.path.exists(logdir):
            os.makedirs(logdir)
    
    def load_model(self, model_path):
        self.model = self.model.load(model_path, env = self.custom_env)
    
    def train_model(
            self, timesteps: int = 10000, log_interval: int = 1,
            reset_num_timesteps: bool = False, progress_bar: bool = False, 
            repeat_times:int = 1, start_from:int = 1, callback : MaybeCallback = None
        ) -> None:
        
        try:
            for repeat_time in range(repeat_times):
                self.model.learn(total_timesteps = timesteps, callback = callback,
                                log_interval = log_interval,
                                reset_num_timesteps = reset_num_timesteps,
                                progress_bar = progress_bar,
                                tb_log_name = self.model_name)
                self.model.save(f"{self.models_dir}/{timesteps*(repeat_time
                                                                 + start_from)}")

            return evaluate_policy(self.model, self.custom_env,
                                    n_eval_episodes=5, deterministic=True)
        finally:
            self.custom_env.close()
    
    def test_model(
            self, total_steps: int = 10000, episodes : int = 100,
            deterministic: bool = True
        ) -> None:

        vec_env = self.model.get_env()
        obs = vec_env.reset()

        total_reward = 0
        curr_step = 0
        for _ in range(episodes):
            done = False
            while not done:
                action, _state = self.model.predict(obs, deterministic=deterministic)
                obs, rewards, done, info = vec_env.step(action)

                total_reward += rewards
                curr_step += 1
            wandb.log({
                "eval/total_reward": total_reward
            })
            total_reward = 0  # Reset reward for the new episode
            if curr_step >= total_steps:
                break
    
    def alternate_train_test(
            self, iterations : int = 100, train_steps : int = 10000, 
            train_repeat_times : int = 1, test_steps : int = 10000, 
            test_episodes : int = 100, callback : MaybeCallback = None
        ) -> None:

        for iteration in range(iterations):
            self.train_model(
                timesteps=train_steps, repeat_times=train_repeat_times,
                start_from=iteration*train_repeat_times +1, callback=callback
            )
            self.test_model(total_steps=test_steps, episodes = test_episodes)
    