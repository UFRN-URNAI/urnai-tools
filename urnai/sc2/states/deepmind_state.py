import numpy as np
from pysc2.lib import features

from urnai.states.state_base import StateBase


class DeepmindState(StateBase):

    def __init__(self):
        self.reset()

    def reset(self):
        self._state = None
        self._dimension = None
    
    @property
    def state(self):
        return self._state

    @property
    def dimension(self):
        screen_dims = self._state["screen"].shape
        minimap_dims = self._state["minimap"].shape
        non_spatial_dims = self._state["non_spatial"].shape
        return (screen_dims, minimap_dims, non_spatial_dims)

    def update(self, obs):
        """
        Pré-processa a observação do PySC2 de acordo com as especificações.
        Retorna um dicionário contendo os tensores para screen, minimap e player.
        """

        processed_screen = self.process_screen(obs)
        processed_minimap = self.process_minimap(obs)
        processed_non_spatial = self.process_non_spatial(obs)

        self._state = {
            "screen": processed_screen,   # [C', H, W]
            "minimap": processed_minimap, # [C'', H, W]
            "non_spatial": processed_non_spatial    # [n_features]
        }
        return self._state

    def process_screen(self, obs):
        screen = obs["feature_screen"]
        screen_processed = []
        for i, spec in enumerate(features.SCREEN_FEATURES):
            channel = screen[i].astype(np.float32)
            processed_channel = self.process_channel(channel, spec)
            screen_processed.append(processed_channel)
        screen_processed = np.concatenate(screen_processed, axis=0)  # [C', H, W]
        return screen_processed
    
    def process_minimap(self, obs):
        minimap = obs["feature_minimap"]
        minimap_processed = []
        for i, spec in enumerate(features.MINIMAP_FEATURES):
            channel = minimap[i].astype(np.float32)
            processed_channel = self.process_channel(channel, spec)
            minimap_processed.append(processed_channel)
        minimap_processed = np.concatenate(minimap_processed, axis=0)
        return minimap_processed

    def process_channel(self, channel, spec):
        if spec.type == features.FeatureType.CATEGORICAL:
            one_hot = np.eye(spec.scale, dtype=np.float32)[channel]
            one_hot = np.transpose(one_hot, (2, 0, 1))
            return one_hot
        else:
            channel = np.log1p(channel)
            return channel[None, :, :]

    def process_non_spatial(self, obs):
        player = obs["player"]
        player_processed = np.log1p(player.astype(np.float32))
        return player_processed