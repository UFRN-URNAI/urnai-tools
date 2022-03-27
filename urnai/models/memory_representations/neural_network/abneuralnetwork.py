from abc import abstractmethod

from models.memory_representations.base.abmem import ABMemoryRepresentation
from models.model_builder import ModelBuilder
from urnai.utils.error import IncoherentBuildModelError, UnsupportedBuildModelLayerTypeError
from urnai.utils.reporter import Reporter as rp


class ABNeuralNetwork(ABMemoryRepresentation):
    """
    Base Class for a Neural Network.

    This class inherits from ABMemoryRepresentation, so it already has
    abstract methods for updating the NN and for predicting an output.

    Therefore, this class just implements the make_model method, using a python
    dict as a base to dinamically build a Neural Network. For that, it uses
    abstract classes that add Neural Network Layers, such as add_input_layer(),
    add_output_layer(), add_fully_connected_layer(), etc.
    """

    def make_model(self):
        self.model = self.create_base_model()

        if self.build_model[0]['type'] == ModelBuilder.LAYER_INPUT \
                and self.build_model[-1]['type'] == ModelBuilder.LAYER_OUTPUT:
            self.build_model[0]['shape'] = self.state_input_shape
        elif self.build_model[0]['type'] == ModelBuilder.LAYER_CONVOLUTIONAL and \
                self.build_model[-1]['type'] == ModelBuilder.LAYER_OUTPUT:
            self.build_model[0]['input_shape'] = self.state_input_shape
        else:
            raise IncoherentBuildModelError(
                'The first layer type should be {} and the last one type should be {}'.format(
                    ModelBuilder.LAYER_INPUT, ModelBuilder.LAYER_OUTPUT))

        self.build_model[-1]['length'] = self.action_output_size

        for idx, (layer_model) in enumerate(self.build_model):
            if layer_model['type'] == ModelBuilder.LAYER_INPUT:
                if idx == 0:
                    self.add_input_layer(idx)
                else:
                    raise IncoherentBuildModelError(
                        'The first layer type should be {}'.format(ModelBuilder.LAYER_INPUT))
            elif layer_model['type'] == ModelBuilder.LAYER_OUTPUT:
                if (idx + 1) == len(self.build_model):
                    self.add_output_layer(idx)
                else:
                    raise IncoherentBuildModelError(
                        'The last layer type should be {}'.format(ModelBuilder.LAYER_OUTPUT))
            elif layer_model['type'] == ModelBuilder.LAYER_FULLY_CONNECTED:
                self.add_fully_connected_layer(idx)
            elif layer_model['type'] == ModelBuilder.LAYER_CONVOLUTIONAL:
                self.add_convolutional_layer(idx)
            elif layer_model['type'] == ModelBuilder.LAYER_MAXPOOLING:
                self.add_maxpooling_layer(idx)
            elif layer_model['type'] == ModelBuilder.LAYER_FLATTEN:
                self.add_flatten_layer(idx)

    @abstractmethod
    def create_base_model(self) -> None:
        """
        This method is used to instantiate a Generic NN model and return it.
        This is necessary because the instantiation of NN models differs from tensorflow.keras
        to Pytorch, so with this method we can separate the instantion of the model
        from its construction.
        """
        ...

    @abstractmethod
    def add_input_layer(self, idx) -> None:
        ...

    @abstractmethod
    def add_output_layer(self, idx) -> None:
        ...

    @abstractmethod
    def copy_model_weights(self, model_to_copy) -> None:
        ...

    def get_device(self):
        rp.report('THIS CLASS DOES NOT YET SUPPORT RETURN THE DEVICE NAME.')

    def add_fully_connected_layer(self, idx):
        raise UnsupportedBuildModelLayerTypeError(
            'Fully-connected layer is not supported by {}.'.format(self.__class__.__name__))

    def add_convolutional_layer(self, idx):
        raise UnsupportedBuildModelLayerTypeError(
            'Convolutional layer is not supported by {}.'.format(self.__class__.__name__))

    def add_maxpooling_layer(self, idx):
        raise UnsupportedBuildModelLayerTypeError(
            'MaxPooling layer is not supported by {}.'.format(self.__class__.__name__))

    def add_flatten_layer(self, idx):
        raise UnsupportedBuildModelLayerTypeError(
            'Flatten layer is not supported by {}.'.format(self.__class__.__name__))
