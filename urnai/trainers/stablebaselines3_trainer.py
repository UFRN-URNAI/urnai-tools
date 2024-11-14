import os

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback


class SB3Trainer:
    def __init__(self, custom_env, models_dir, logdir, model : BaseAlgorithm):
        self.custom_env = custom_env
        self.models_dir = models_dir
        self.model = model

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if not os.path.exists(logdir):
            os.makedirs(logdir)
    
    def load_model(self, model_path):
        self.model = self.model.load(model_path, env = self.custom_env)
    
    def train_model(self, timesteps: int = 10000, callback: MaybeCallback = None,
                    log_interval: int = 100, tb_log_name: str = "run",
                    reset_num_timesteps: bool = True, progress_bar: bool = False,
                    repeat_times: int = 1):
        for repeat_time in range(repeat_times):
            self.model.learn(total_timesteps = timesteps, callback = callback,
                             log_interval = log_interval, tb_log_name = tb_log_name,
                             reset_num_timesteps = reset_num_timesteps,
                             progress_bar = progress_bar)
            self.model.save(f"{self.models_dir}/{timesteps*(repeat_time + 1)}")
    
    def test_model(self, total_steps: int = 10000, deterministic: bool = True):
        vec_env = self.model.get_env()
        obs = vec_env.reset()

        total_episodes = 0
        total_reward = 0

        for _ in range(total_steps):
            action, _state = self.model.predict(obs, deterministic=deterministic)
            obs, rewards, done, info = vec_env.step(action)

            total_reward += rewards
            if done:
                total_episodes += 1
                total_reward = 0  # Reset reward for the new episode
    