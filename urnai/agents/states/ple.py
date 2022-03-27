import copy

import numpy as np

from .abstate import StateBuilder


class FlappyBirdState(StateBuilder):

    def __init__(self):
        self.range_per_feature = {
            'next_next_pipe_bottom_y': 40,
            'next_next_pipe_dist_to_player': 512,
            'next_next_pipe_top_y': 40,
            'next_pipe_bottom_y': 20,
            'next_pipe_dist_to_player': 20,
            'next_pipe_top_y': 20,
            'player_vel': 4,
            'player_y': 16,
        }

    def build_state(self, obs):
        # instead of using absolute position of pipe, use relative position
        state = copy.deepcopy(obs)
        state['next_next_pipe_bottom_y'] -= state['player_y']
        state['next_next_pipe_top_y'] -= state['player_y']
        state['next_pipe_bottom_y'] -= state['player_y']
        state['next_pipe_top_y'] -= state['player_y']

        # sort to make list converted from dict ordered in alphabet order
        state = [v / self.range_per_feature[k] for k, v in state.items()]
        state = np.array(state)
        state = state.reshape(1, self.get_state_dim())

        return state

    def get_state_dim(self):
        return 8
