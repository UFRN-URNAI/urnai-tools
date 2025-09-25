from urnai.models.model_base import ModelBase
from urnai.sc2.models.atarinet_neural_network import AtariNetNeuralNetwork


class AtariNetModel(ModelBase):
    def __init__(self):
        super().__init__()

        self.neural_network = AtariNetNeuralNetwork(
            input_channels_screen = 19, #TODO: remove magic numbers
            input_channels_minimap = 9,
            input_channels_nonspatial = 11,
            height = 64, width = 64
        )

    def learn(self, current_state, action, reward, next_state, done) -> None:
        ...

    def predict(self, state) -> int:
        ...

