import gym
from gym.envs.registration import register
from urnai.agents.actions.gym_wrapper import GymWrapper

from .base.abenv import Env

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)


class GymEnv(Env):
    def __init__(self, id, render=False, reset_done=True, num_steps=100):
        super().__init__(id, render, reset_done)
        self.num_steps = num_steps

        self.start()

    def start(self):
        if not self.env_instance:
            self.env_instance = gym.make(self.id)

    def step(self, action):
        obs, reward, done, _ = self.env_instance.step(action)
        if self.render:
            self.env_instance.render()
        return obs, reward, done

    def reset(self):
        return self.env_instance.reset()

    def close(self):
        self.env_instance.close()

    def restart(self):
        self.close()
        self.reset()

    def get_action_wrapper(self):
        if self.env_instance is not None:
            return GymWrapper(self.env_instance.action_space.n)
