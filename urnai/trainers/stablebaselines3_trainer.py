import os

from stable_baselines3.common.base_class import BaseAlgorithm

import wandb
from urnai.environments.stablebaselines3.custom_env import CustomEnv
from wandb.integration.sb3 import WandbCallback


class SB3Trainer:
    def __init__(self, custom_env : CustomEnv, models_dir : str, logdir : str, 
                 model : BaseAlgorithm, model_name : str, 
                 wandb_project : str= "default_project", wandb_conf_dict : dict = None):
        self.custom_env = custom_env
        self.models_dir = models_dir
        self.model = model
        self.model_name = model_name
        self.wandb_project = wandb_project
        self.wandb_conf_dict = wandb_conf_dict

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if not os.path.exists(logdir):
            os.makedirs(logdir)
    
    def load_model(self, model_path):
        self.model = self.model.load(model_path, env = self.custom_env)
    
    def train_model(self, timesteps: int = 10000, log_interval: int = 1,
                    reset_num_timesteps: bool = False, progress_bar: bool = False, 
                    repeat_times:int = 1, start_from:int = 1, train_run_id:str = None,
                    ) -> str:
        
        wandb_run = wandb.init(
            project=self.wandb_project,
            config=self.wandb_conf_dict,
            name=f"train_{self.model_name}",
            group="train",
            sync_tensorboard=True,
            resume="must" if train_run_id else None,
            id=train_run_id
        )
        
        for repeat_time in range(repeat_times):
            self.model.learn(total_timesteps = timesteps, callback = WandbCallback(),
                             log_interval = log_interval,
                             reset_num_timesteps = reset_num_timesteps,
                             progress_bar = progress_bar,
                             tb_log_name = self.model_name)
            self.model.save(f"{self.models_dir}/{timesteps*(repeat_time + start_from)}")
        
        wandb_run.finish()
        return wandb_run.id
    
    def test_model(self, total_steps: int = 10000, 
        deterministic: bool = True, 
        test_run_id : str = None
        ) -> str:
        wandb_run = wandb.init(
            project=self.wandb_project,
            name=f"test_{self.model_name}",
            group="test", 
            resume="must" if test_run_id else None,
            id=test_run_id
        )

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
                wandb.log({
                    "test/total_reward": total_reward,
                    "test/episode": total_episodes,
                })
                total_reward = 0  # Reset reward for the new episode
        
        wandb_run.finish()
        return wandb_run.id
    
    def alternate_train_test(self, iterations : int = 100, train_steps : int = 10000, 
                             train_repeat_times : int = 1, test_steps : int = 10000, 
                             train_run_id = None, test_run_id = None
                             ) -> None:

        for iteration in range(iterations):
            train_run_id = self.train_model(timesteps=train_steps, 
                                            repeat_times=train_repeat_times, 
                                            train_run_id=train_run_id, 
                                            start_from=iteration*train_repeat_times +1)
            test_run_id= self.test_model(total_steps=test_steps,test_run_id=test_run_id)
    