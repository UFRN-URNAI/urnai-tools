from typing import Tuple
from .base.abenv import Env
import numpy as np
import random
import time

class ClickGameEnv(Env):
    def __init__(self, render=False, board_shape=[10, 10], max_steps=200, use_marine=False, marine_value=-1):
        self.pickle_black_list = ["env_instance", "render", "board_shape", "max_steps", "curr_steps", "x", "y", "use_marine"]
        self.env_instance = None
        self.render = render
        self.board_shape = board_shape
        self.max_steps = max_steps
        self.use_marine = use_marine
        self.marine_value = marine_value
        self.curr_steps = 0
        self.x = 0
        self.y = 0
        self.start()

    def start(self):
        if self.env_instance is None:
            board = np.zeros((self.board_shape[1], self.board_shape[0], 1))
            self.x = random.randint(0, self.board_shape[0]-1)
            self.y = random.randint(0, self.board_shape[1]-1)
            board[self.y][self.x] = 1
            if self.use_marine:
                marine = [random.randint(0, self.board_shape[0]-1), random.randint(0, self.board_shape[1]-1)]
                board[marine[1]][marine[0]] = self.marine_value
            self.env_instance = [board, self.x, self.y]

    def step(self, action):
        x, y = action
        reward = 0

        if self.render:
            np.set_printoptions(threshold=np.inf)
            print("Rodada {}".format(self.curr_steps))
            print("State: x: {}, y: {}".format(self.x, self.y))
            print("Ação : x: {}, y: {}".format(x, y))
            print(self.env_instance)
            np.set_printoptions(threshold=1000)
            time.sleep(1)

        if x == self.x and y == self.y:
            board = np.zeros((self.board_shape[1], self.board_shape[0], 1))
            self.x = random.randint(0, self.board_shape[0]-1)
            self.y = random.randint(0, self.board_shape[1]-1)
            board[self.y][self.x] = 1
            if self.use_marine:
                marine = [random.randint(0, self.board_shape[0]-1), random.randint(0, self.board_shape[1]-1)]
                board[marine[1]][marine[0]] = self.marine_value
            self.env_instance = [board, self.x, self.y]
            reward = 1

        self.curr_steps += 1
        
        done = True if self.curr_steps == self.max_steps else False
        
        return self.env_instance, reward, done

    def reset(self):
        board = np.zeros((self.board_shape[1], self.board_shape[0], 1))
        self.x = random.randint(0, self.board_shape[0]-1)
        self.y = random.randint(0, self.board_shape[1]-1)
        board[self.y][self.x] = 1
        if self.use_marine:
            marine = [random.randint(0, self.board_shape[0]-1), random.randint(0, self.board_shape[1]-1)]
            board[marine[1]][marine[0]] = self.marine_value
        self.env_instance = [board, self.x, self.y]
        self.curr_steps = 0
        return self.env_instance

    def close(self):
        pass
