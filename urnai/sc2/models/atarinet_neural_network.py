import torch
import torch.nn.functional as F
from torch import nn


class AtariNetNeuralNetwork(nn.Module):
    def __init__(self,
                 input_channels_screen,
                 input_channels_minimap,
                 input_channels_nonspatial,
                 height, width,
                 action_space_info):
        super().__init__()
        """
        Atari-net Neural Network as described in
        "StarCraft II: A New Challenge for Reinforcement Learning"
        """

        self.nonspatial_dense = nn.Linear(input_channels_nonspatial, 32)

        self.screen_conv1 = nn.Conv2d(
            in_channels=input_channels_screen,
            out_channels=16,
            kernel_size=8,
            stride=4,
            padding=0
        )
        self.screen_conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=0
        )

        self.minimap_conv1 = nn.Conv2d(
            in_channels=input_channels_minimap,
            out_channels=16,
            kernel_size=8,
            stride=4,
            padding=0
        )
        self.minimap_conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=0
        )

        with torch.no_grad():
            dummy_screen = torch.zeros(1, input_channels_screen, height, width)
            dummy_minimap = torch.zeros(1, input_channels_minimap, height, width)
            dummy_nonspatial = torch.zeros(1, input_channels_nonspatial)

            s = self._forward_screen(dummy_screen)
            m = self._forward_minimap(dummy_minimap)
            n = self._forward_nonspatial(dummy_nonspatial)

            combined_dim = s.shape[1] + m.shape[1] + n.shape[1]

        out_combined_dense = 256
        self.combined_dense = nn.Linear(combined_dim, out_combined_dense)

        self.function_identifier = nn.Linear(
            in_features=out_combined_dense,
            out_features=len(action_space_info["functions"]),
        )

        # For now, ignore anything below here

        def _fix_map_actions_range(action_type): # TODO: remove later
            if action_type.name in ['screen', 'minimap', 'screen2']:
                return (64, 64)
            return action_type.sizes

        self.function_arg = {}
        for action_type in action_space_info["types"]:
            self.function_arg[action_type.name] = {}

            sizes = _fix_map_actions_range(action_type)

            for dim_index, action_size in enumerate(sizes):
                self.function_arg[action_type.name][dim_index] = nn.Linear(
                    in_features=out_combined_dense,
                    out_features=action_size,
                )

    def forward(self, x):
        inputs_screen, inputs_minimap, inputs_nonspatial = x

        nonspatial = self._forward_nonspatial(inputs_nonspatial)
        screen = self._forward_screen(inputs_screen)
        minimap = self._forward_minimap(inputs_minimap)

        combined = torch.cat([nonspatial, screen, minimap], dim=1)
        combined = F.relu(self.combined_dense(combined))

        func_id = self.function_identifier(combined)

        # TODO: return argument values
        """
        argument_values = {}
        for arg_name in self.function_arg:
            argument_values[arg_name] = {}

            for dim_index in self.function_arg[arg_name]:
                arg_value = self.function_arg[arg_name][dim_index](combined)
                argument_values[arg_name][dim_index] = arg_value
        """

        return func_id

    def _forward_screen(self, inputs_screen):
        screen = F.relu(self.screen_conv1(inputs_screen))
        screen = F.relu(self.screen_conv2(screen))
        screen = torch.flatten(screen, start_dim=1)

        return screen
    
    def _forward_minimap(self, inputs_minimap):
        minimap = F.relu(self.minimap_conv1(inputs_minimap))
        minimap = F.relu(self.minimap_conv2(minimap))
        minimap = torch.flatten(minimap, start_dim=1)

        return minimap
    
    def _forward_nonspatial(self, inputs_nonspatial):
        nonspatial = torch.tanh(self.nonspatial_dense(inputs_nonspatial))

        return nonspatial