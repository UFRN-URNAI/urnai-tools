import numpy as np
import torch
from pysc2.lib import actions

from urnai.models.model_base import ModelBase
from urnai.sc2.models.atarinet_neural_network import AtariNetNeuralNetwork


class AtariNetModel(ModelBase):
    def __init__(self):
        super().__init__()

        #TODO: remove magic numbers (they must be sourced from the state)
        self.neural_network = AtariNetNeuralNetwork(
            input_channels_screen = 19,
            input_channels_minimap = 9,
            input_channels_nonspatial = 11,
            height = 64, width = 64
        )

    def learn(self, current_state, action, reward, next_state, done) -> None:
        ...

    def predict(self, state) -> int: #TODO: change the type

        def to_input(state_data):
            return (torch.from_numpy(state_data)).unsqueeze(0)

        inputs_screen = to_input(state["screen"])
        inputs_minimap = to_input(state["minimap"])
        inputs_nonspatial = to_input(state["non_spatial"])

        function_id_softmax, arguments_softmax = self.neural_network(
            (inputs_screen, inputs_minimap, inputs_nonspatial)
        )

        function_id = np.argmax(function_id_softmax.detach().numpy())

        arguments = []
        for arg_name in list(actions.FUNCTION_TYPES)[function_id]:
            for dim in arguments_softmax[arg_name]:
                arg_value = np.argmax(arguments_softmax[arg_name][dim].detach().numpy())
                arguments.append(arg_value)

        return function_id, arguments


