from ple import PLE

from .base.abenv import Env


class PLEEnv(Env):
    def __init__(self, game, _id, render=True, reset_done=True, num_steps=100):
        super().__init__(_id, render, reset_done)
        self.num_steps = num_steps
        self.game = game
        self.start()

    def start(self):
        if not self.env_instance:
            self.env_instance = PLE(self.game, fps=30, display_screen=self.render)
            self.env_instance.init()

    def step(self, action):
        reward = self.env_instance.act(action)
        obs = self.env_instance.getGameState()
        done = self.env_instance.game_over()
        return obs, reward, done

    def reset(self):
        self.env_instance.reset_game()
        obs = self.env_instance.getGameState()
        return obs

    def close(self):
        pass

    def restart(self):
        self.close()
        self.reset()
