import os
import sys

import numpy as np
import torch
from absl import app
from pysc2.env import sc2_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from urnai.sc2.actions.collectables import CollectablesActionSpace
from urnai.sc2.environments.sc2environment import SC2Env
from urnai.sc2.models.atarinet_model import AtariNetModel
from urnai.sc2.rewards.collectables import CollectablesReward


def make_env():

    class QuickEnv:
        def __init__(self):
            players = [sc2_env.Agent(sc2_env.Race.terran)]
            self._env = SC2Env(map_name='CollectMineralShards', visualize=False, 
                step_mul=16, players=players)

            self._state = ...#DeepMindState()
            self._action_space = CollectablesActionSpace()
            self._reward = CollectablesReward()

            self._obs = self._env.reset()
        
        def step(self, action):
            action = self._action_space.get_action(action, self._obs)

            obs, reward, terminated, truncated = self._env.step(action)

            self._obs = obs
            obs = self._state.update(self._obs)
            reward = self._reward.get(self._obs, reward, terminated, truncated)
            return obs, reward, terminated or truncated

    env = QuickEnv()
    
    return env

def main(_):
    input_channels_screen = 19
    input_channels_minimap = 9
    input_channels_nonspatial = 11

    height, width = 64, 64

    inputs_screen = np.array(torch.randn(input_channels_screen, height, width))
    inputs_minimap = np.array(torch.randn(input_channels_minimap, height, width))
    inputs_nonspatial = np.array(torch.randn(input_channels_nonspatial))

    state = {
        "screen" : inputs_screen,
        "minimap" : inputs_minimap,
        "non_spatial" : inputs_nonspatial
    }

    model = AtariNetModel(map_name='CollectMineralShards')
    env = make_env()
    
    for _ in range(100):
        function_id, args = model.predict(state)
        action = function_id #TODO: Make ActionSpace receive action arguments

        next_state, reward, done = env.step(action)

        model.learn(state, action, reward, next_state, done)

        print(reward)

        if done:
            break

if __name__ == '__main__':
    app.run(main)