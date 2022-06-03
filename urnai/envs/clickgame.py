from typing import Tuple
from .base.abenv import Env
import numpy as np
import random

class ClickGameEnv(Env):
    def __init__(self, render=False, board_shape=[10, 10], max_steps=200):
        self.render = render
        self.board_shape = board_shape
        self.max_steps = max_steps
        self.curr_steps = 0
        self.x = 0
        self.y = 0
        self.start()

    def start(self):
        board = np.zeros((self.board_shape[0], self.board_shape[1]))
        self.x = random.randint(0, self.board_shape[0]-1)
        self.y = random.randint(0, self.board_shape[1]-1)
        board[self.y][self.x] = 1
        self.env_instance = board

    def step(self, action):
        x, y = action
        reward = 0
        if x == self.x and y == self.y:
            self.env_instance[self.y][self.x] = 0
            self.x = random.randint(0, self.board_shape[0]-1)
            self.y = random.randint(0, self.board_shape[1]-1)
            self.env_instance[self.y][self.x] = 1
            reward = 1

        self.curr_steps += 1
        
        done = True if self.curr_steps == self.max_steps else False

        # if self.render:
        #     np.set_printoptions(threshold=np.inf)
        #     print()
        #     print(self.env_instance)
        #     np.set_printoptions(threshold=1000)

        return self.env_instance, reward, done

    def reset(self):
        self.start()
        self.curr_steps = 0
        return self.env_instance

    def close(self):
        self.env_instance.close()
