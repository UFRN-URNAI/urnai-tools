import numpy as np
import torch

from urnai.sc2.models.atarinet_neural_network import AtariNetNeuralNetwork


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

""" Uses DeepmindState
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from urnai.sc2.states.deepmind_state import DeepmindState

state = DeepmindState()
obs = np.load("saves/logs/example_state.npz")
input_to_nn = state.update(obs)
for a in input_to_nn:
    print(a, input_to_nn[a].shape)

input_channels_screen = input_to_nn["screen"].shape[0]
input_channels_minimap = input_to_nn["minimap"].shape[0]
input_channels_nonspatial = input_to_nn["non_spatial"].shape[0]

inputs_screen = (torch.from_numpy(input_to_nn["screen"])).unsqueeze(0)
inputs_minimap = (torch.from_numpy(input_to_nn["minimap"])).unsqueeze(0)
inputs_nonspatial = (torch.from_numpy(input_to_nn["non_spatial"])).unsqueeze(0)

"""

input_channels_screen = 19
input_channels_minimap = 9
input_channels_nonspatial = 11

batch_size = 2
height, width = 64, 64

inputs_screen = torch.randn(batch_size, input_channels_screen, height, width)
inputs_minimap = torch.randn(batch_size, input_channels_minimap, height, width)
inputs_nonspatial = torch.randn(batch_size, input_channels_nonspatial)


model = AtariNetNeuralNetwork(
    input_channels_screen, input_channels_minimap, input_channels_nonspatial,
    height, width)
logits = model((inputs_screen, inputs_minimap, inputs_nonspatial))
print(logits.shape)