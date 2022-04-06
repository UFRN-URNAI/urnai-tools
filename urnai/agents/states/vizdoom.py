from collections import deque

import numpy as np

from .abstate import StateBuilder


class DefaultVizDoomState(StateBuilder):
    def build_state(self, obs):
        return obs

    def get_state_dim(self, obs):
        return len(obs)


class VizDoomHealthGatheringState(StateBuilder):

    def __init__(self, screen_width, screen_height, slices=1, lib='keras'):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.state_dim = [self.screen_width, self.screen_height]
        self.slices = slices
        self.lib = lib

        lst = []
        for i in range(self.slices):
            lst.append(np.zeros((self.screen_height, self.screen_width), dtype=np.int))

        self.stacked_frames = deque(lst, maxlen=self.slices)

    def build_state(self, obs):
        self.stacked_frames.append(obs.screen_buffer)
        if self.lib == 'keras':
            arr = np.stack(self.stacked_frames)
            arr = arr.reshape(self.slices, self.screen_height, self.screen_width, 1)
            return arr

    def get_state_dim(self):
        return self.state_dim
