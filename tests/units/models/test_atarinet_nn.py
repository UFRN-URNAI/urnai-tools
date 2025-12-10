import unittest
import torch

from urnai.sc2.models.atarinet_neural_network import AtariNetNeuralNetwork


def make_dummy_action_space():
    class A:
        def __init__(self, name, sizes):
            self.name = name
            self.sizes = sizes

    return {
        "functions": ["move", "attack", "idle"],
        "types": [
            A("screen", [64, 64]),
            A("minimap", [64, 64]),
            A("nonspatial", [11]),
        ],
    }

class TestAtariNetNeuralNetwork(unittest.TestCase):

    def setUp(self):
        self.input_channels_screen = 12
        self.input_channels_minimap = 2
        self.input_channels_nonspatial = 11
        self.height = 64
        self.width = 64

        self.action_info = make_dummy_action_space()

        self.net = AtariNetNeuralNetwork(
            self.input_channels_screen,
            self.input_channels_minimap,
            self.input_channels_nonspatial,
            self.height,
            self.width,
            self.action_info
        )

    def test_forward_screen_shape(self):
        # GIVEN
        x = torch.zeros(1, self.input_channels_screen, 64, 64)

        # WHEN
        out = self.net._forward_screen(x)

        # THEN
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 1152)

    def test_forward_minimap_shape(self):
        # GIVEN
        x = torch.zeros(1, self.input_channels_minimap, 64, 64)

        # WHEN
        out = self.net._forward_minimap(x)

        # THEN
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 1152)

    def test_forward_nonspatial_shape_and_range(self):
        # GIVEN
        x = torch.rand(1, self.input_channels_nonspatial)

        # WHEN
        out = self.net._forward_nonspatial(x)

        # THEN
        self.assertEqual(out.shape, (1, 32))
        self.assertTrue(torch.all(out <= 1.0))
        self.assertTrue(torch.all(out >= -1.0))  # tanh range

    def test_forward_output_shape(self):
        # GIVEN
        screen = torch.zeros(1, self.input_channels_screen, 64, 64)
        minimap = torch.zeros(1, self.input_channels_minimap, 64, 64)
        nonspatial = torch.zeros(1, self.input_channels_nonspatial)

        # WHEN
        out = self.net((screen, minimap, nonspatial))

        # THEN
        expected_outputs = len(self.action_info["functions"])
        self.assertEqual(out.shape, (1, expected_outputs))

    def test_forward_no_nans(self):
        # GIVEN
        screen = torch.randn(1, self.input_channels_screen, 64, 64)
        minimap = torch.randn(1, self.input_channels_minimap, 64, 64)
        nonspatial = torch.randn(1, self.input_channels_nonspatial)

        # WHEN
        out = self.net((screen, minimap, nonspatial))

        # THEN
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_forward_deterministic_with_seed(self):
        # GIVEN
        torch.manual_seed(123)
        net1 = AtariNetNeuralNetwork(
            self.input_channels_screen,
            self.input_channels_minimap,
            self.input_channels_nonspatial,
            self.height,
            self.width,
            self.action_info
        )

        torch.manual_seed(123)
        net2 = AtariNetNeuralNetwork(
            self.input_channels_screen,
            self.input_channels_minimap,
            self.input_channels_nonspatial,
            self.height,
            self.width,
            self.action_info
        )

        screen = torch.randn(1, self.input_channels_screen, 64, 64)
        minimap = torch.randn(1, self.input_channels_minimap, 64, 64)
        nonspatial = torch.randn(1, self.input_channels_nonspatial)

        # WHEN
        out1 = net1((screen, minimap, nonspatial))
        out2 = net2((screen, minimap, nonspatial))

        # THEN
        self.assertTrue(torch.allclose(out1, out2))