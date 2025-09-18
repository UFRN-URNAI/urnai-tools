from typing import Any

import numpy as np
import torch
import torch.nn as nn
from pysc2.lib import features

from urnai.states.state_base import StateBase


class DeepmindState(StateBase):
    """State processor used by DeepMind for StarCraft II observations."""

    def __init__(self, categorical_conv_channels: int = 8):
        self.categorical_out_conv_channels = categorical_conv_channels
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
        processed_screen = self.process_screen(obs)
        processed_minimap = self.process_minimap(obs)
        processed_non_spatial = self.process_non_spatial(obs)

        self._state = {
            "screen": processed_screen,       # [C', H, W]
            "minimap": processed_minimap,     # [C'', H, W]
            "non_spatial": processed_non_spatial    # [n_features]
        }
        return self._state

    def _separate_raw_channels_by_type(
        self,
        spatial_data: np.ndarray,
        feature_specs: list
    ) -> tuple[
        list[tuple[np.ndarray, features.Feature]],
        list[tuple[np.ndarray, features.Feature]]
    ]:
        categorical_items = []
        numerical_items = []
        
        for i, spec in enumerate(feature_specs):
            channel = spatial_data[i].astype(np.float32)
            
            if spec.type == features.FeatureType.CATEGORICAL:
                categorical_items.append((channel, spec))
            else:
                numerical_items.append((channel, spec))
                
        return categorical_items, numerical_items
    
    def _process_categorical_channels(
        self,
        categorical_items: list[tuple[np.ndarray, features.Feature]]
    ) -> np.ndarray:
        if not categorical_items:
            return None
            
        # Apply one-hot encoding to all categorical channels
        one_hot_channels = []
        for channel, spec in categorical_items:
            # Convert to integer indices for one-hot encoding
            channel_int = channel.astype(np.int32)
            one_hot = np.eye(spec.scale, dtype=np.float32)[channel_int]
            one_hot = np.transpose(one_hot, (2, 0, 1))  # [scale, H, W]
            one_hot_channels.append(one_hot)
            
        # Concatenate all one-hot encoded channels
        cat_combined = np.concatenate(one_hot_channels, axis=0)
        
        # Apply conv1x1 to reduce dimensions
        return self.conv1x1(
            cat_combined.shape[0], 
            self.categorical_out_conv_channels, 
            cat_combined
        )
    
    def _process_numerical_channels(
        self,
        numerical_items: list[tuple[np.ndarray, features.Feature]]
    ) -> np.ndarray:
        if not numerical_items:
            return None
            
        processed_channels = []
        for channel, _ in numerical_items:
            # Apply log transformation and add channel dimension
            processed_channel = np.log1p(channel)
            processed_channel = processed_channel[None, :, :]  # Add channel dimension
            processed_channels.append(processed_channel)

        return np.concatenate(processed_channels, axis=0)
    
    def _combine_all_channels(
        self,
        processed_categorical: np.ndarray,
        processed_numerical: np.ndarray,
        spatial_shape: tuple[int, int]
    ) -> np.ndarray:
        all_channels = []

        if processed_categorical is not None and processed_categorical.size > 0:
            all_channels.append(processed_categorical)

        if processed_numerical is not None and processed_numerical.size > 0:
            all_channels.append(processed_numerical)

        if all_channels:
            return np.concatenate(all_channels, axis=0)
        else:
            # Return empty array with correct shape if no channels
            height, width = spatial_shape
            return np.zeros((0, height, width), dtype=np.float32)

    def _process_spatial_features(
        self,
        spatial_data: np.ndarray,
        feature_specs: list
    ) -> np.ndarray:
        categorical_items, numerical_items = self._separate_raw_channels_by_type(
            spatial_data, feature_specs
        )
        
        processed_categorical = self._process_categorical_channels(categorical_items)
        
        processed_numerical = self._process_numerical_channels(numerical_items)

        spatial_shape = (spatial_data.shape[1], spatial_data.shape[2])
        return self._combine_all_channels(
            processed_categorical, processed_numerical, spatial_shape
        )

    def process_screen(self, obs):
        """Process screen features using the generic spatial processing method."""
        return self._process_spatial_features(
            obs["feature_screen"], 
            features.SCREEN_FEATURES
        )

    def process_minimap(self, obs):
        """Process minimap features using the generic spatial processing method."""
        return self._process_spatial_features(
            obs["feature_minimap"], 
            features.MINIMAP_FEATURES
        )

    def conv1x1(
        self,
        in_channels: int,
        out_channels: int,
        channels: np.ndarray
    ) -> np.ndarray:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        return conv(torch.from_numpy(channels)).detach().numpy()

    def process_non_spatial(self, obs: dict[str, Any]) -> np.ndarray:
        player = obs["player"]
        player_processed = np.log1p(player.astype(np.float32))
        return player_processed