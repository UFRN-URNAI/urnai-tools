import unittest
from unittest.mock import Mock, patch

import numpy as np
from pysc2.lib import features

from urnai.sc2.states.deepmind_state import DeepmindState


class TestUtilityMethods(unittest.TestCase):
    """Test low-level utility methods"""
    
    def setUp(self):
        self.state = DeepmindState()
    
    def test_one_hot_encode_channel(self):
        # GIVEN
        height = 3
        width = 3
        num_classes = 4
        channel_value = 2
        channel = np.full((height, width), channel_value, dtype=np.float32)

        # WHEN
        one_hot_encoded = self.state._one_hot_encode_channel(channel, num_classes)

        # THEN
        self.assertEqual(one_hot_encoded.shape, (num_classes, height, width))
        for i in range(num_classes):
            expected_value = 1 if i == channel_value else 0
            np.testing.assert_array_equal(
                one_hot_encoded[i], np.full((height, width), expected_value)
            )
    
    @patch('urnai.sc2.states.deepmind_state.DeepmindState.get_cached_conv1x1')
    def test_conv1x1(self, mock_get_cached_conv1x1):
        # GIVEN
        in_channels = 3
        out_channels = 2
        height = 4
        width = 4
        input_array = np.arange(in_channels * height * width, dtype=np.float32).reshape(
            in_channels, height, width
        )

        expected_output = np.ones((out_channels, height, width))
        mock_conv = Mock()
        mock_conv.return_value.detach.return_value.numpy.return_value = expected_output
        mock_get_cached_conv1x1.return_value = mock_conv

        # WHEN
        result = self.state.conv1x1(in_channels, out_channels, input_array)

        # THEN
        mock_get_cached_conv1x1.assert_called_once_with(in_channels, out_channels)
        np.testing.assert_array_equal(result, expected_output)

    @patch('torch.nn.Conv2d')
    def test_get_cached_conv1x1(self, mock_conv2d_class):
        # GIVEN
        in_channels = 5
        out_channels = 3
        
        # WHEN
        conv1 = self.state.get_cached_conv1x1(in_channels, out_channels)
        conv2 = self.state.get_cached_conv1x1(in_channels, out_channels)  # Second call
        
        # THEN
        self.assertIs(conv1, conv2)  # Same instance should be returned
        mock_conv2d_class.assert_called_once_with(
            in_channels, out_channels, kernel_size=1
        )


class TestChannelProcessing(unittest.TestCase):
    """Test channel separation and processing methods"""
    
    def setUp(self):
        self.state = DeepmindState()
        self.height = 4
        self.width = 4
    
    def test_separate_raw_channels_by_type_minimap(self):
        # GIVEN
        feature_specs = features.MINIMAP_FEATURES
        spatial_data = np.arange(
            len(feature_specs) * self.height * self.width, dtype=np.float32
        ).reshape(len(feature_specs), self.height, self.width)

        # WHEN
        categorical_items, numerical_items = self.state._separate_raw_channels_by_type(
            spatial_data, feature_specs
        )

        # THEN
        categorical_count = sum(
            1 for spec in feature_specs if spec.type == features.FeatureType.CATEGORICAL
        )
        numerical_count = len(feature_specs) - categorical_count
        
        self.assertEqual(len(categorical_items), categorical_count)
        self.assertEqual(len(numerical_items), numerical_count)

    def test_separate_raw_channels_by_type_screen(self):
        # GIVEN
        feature_specs = features.SCREEN_FEATURES
        spatial_data = np.arange(
            len(feature_specs) * self.height * self.width, dtype=np.float32
        ).reshape(len(feature_specs), self.height, self.width)

        # WHEN
        categorical_items, numerical_items = self.state._separate_raw_channels_by_type(
            spatial_data, feature_specs
        )

        # THEN
        categorical_count = sum(
            1 for spec in feature_specs if spec.type == features.FeatureType.CATEGORICAL
        )
        numerical_count = len(feature_specs) - categorical_count
        
        self.assertEqual(len(categorical_items), categorical_count)
        self.assertEqual(len(numerical_items), numerical_count)

    def test_separate_raw_channels_by_type_size_mismatch(self):
        # GIVEN
        feature_specs = [Mock(), Mock(), Mock()]  # 3 specs
        spatial_data = np.arange(
            2 * self.height * self.width, dtype=np.float32
        ).reshape(2, self.height, self.width)  # 2 channels

        # WHEN & THEN
        with self.assertRaises(ValueError) as context:
            self.state._separate_raw_channels_by_type(spatial_data, feature_specs)
        
        self.assertIn("Channel count mismatch", str(context.exception))

    def test_process_numerical_channels(self):
        # GIVEN
        numerical_items = [
            (np.full((self.height, self.width), 10.0, dtype=np.float32), Mock()),
            (np.full((self.height, self.width), 5.0, dtype=np.float32), Mock()),
        ]

        # WHEN
        result = self.state._process_numerical_channels(numerical_items)

        # THEN
        self.assertEqual(result.shape, (2, self.height, self.width))
        expected_0 = np.log1p(10.0)
        expected_1 = np.log1p(5.0)
        np.testing.assert_array_almost_equal(
            result[0], np.full((self.height, self.width), expected_0)
        )
        np.testing.assert_array_almost_equal(
            result[1], np.full((self.height, self.width), expected_1)
        )

    def test_process_empty_numerical_channels(self):
        # GIVEN
        numerical_items = []

        # WHEN
        result = self.state._process_numerical_channels(numerical_items)

        # THEN
        self.assertIsNone(result)

    @patch('urnai.sc2.states.deepmind_state.DeepmindState._one_hot_encode_channel')
    @patch('urnai.sc2.states.deepmind_state.DeepmindState.conv1x1')
    def test_process_categorical_channels(self, mock_conv1x1, mock_one_hot_encode):
        # GIVEN
        conv_output_channels = 4
        state = DeepmindState(categorical_conv_channels=conv_output_channels)
        num_classes_list = [3, 4, 2]
        feature_specs_list = [
            Mock(scale=num_classes_list[0]),
            Mock(scale=num_classes_list[1]),
            Mock(scale=num_classes_list[2])
        ]
        categorical_items = [
            (np.full((self.height, self.width), 0), feature_specs_list[0]),
            (np.full((self.height, self.width), 1), feature_specs_list[1]),
            (np.full((self.height, self.width), 0), feature_specs_list[2])
        ]
        
        one_hot_outputs = [
            np.zeros((num_classes_list[0], self.height, self.width), dtype=np.float32),
            np.zeros((num_classes_list[1], self.height, self.width), dtype=np.float32),
            np.zeros((num_classes_list[2], self.height, self.width), dtype=np.float32)
        ]
        one_hot_outputs[0][0] = 1
        one_hot_outputs[1][1] = 1
        one_hot_outputs[2][0] = 1
        
        mock_one_hot_encode.side_effect = one_hot_outputs
        
        concatenated_one_hot = np.concatenate(one_hot_outputs, axis=0)
        
        conv_output = np.zeros(
            (conv_output_channels, self.height, self.width), dtype=np.float32
        )
        for i in range(conv_output_channels):
            conv_output[i] = i + 1
        
        mock_conv1x1.return_value = conv_output

        # WHEN
        processed_categorical = state._process_categorical_channels(categorical_items)

        # THEN
        self.assertEqual(
            processed_categorical.shape, (conv_output_channels, self.height, self.width)
        )
        np.testing.assert_array_equal(processed_categorical, conv_output)
        
        self.assertEqual(mock_one_hot_encode.call_count, len(categorical_items))
        for i, (expected_channel, spec) in enumerate(categorical_items):
            call_args = mock_one_hot_encode.call_args_list[i][0]
            called_channel = call_args[0]
            called_num_classes = call_args[1]
            
            np.testing.assert_array_equal(called_channel, expected_channel)
            self.assertEqual(called_num_classes, spec.scale)
        
        # Check conv1x1 was called once with correct arguments
        mock_conv1x1.assert_called_once()
        call_args = mock_conv1x1.call_args[0]
        self.assertEqual(call_args[0], sum(num_classes_list))
        self.assertEqual(call_args[1], state.categorical_out_conv_channels)
        np.testing.assert_array_equal(call_args[2], concatenated_one_hot)

    def test_process_categorical_channels_empty(self):
        # GIVEN
        categorical_items = []

        # WHEN
        result = self.state._process_categorical_channels(categorical_items)

        # THEN
        self.assertIsNone(result)

    def test_combine_all_channels(self):
        # GIVEN
        processed_categorical = np.ones((2, self.height, self.width), dtype=np.float32)
        processed_numerical = np.full(
            (3, self.height, self.width), 2.0, dtype=np.float32
        )
        spatial_shape = (self.height, self.width)

        # WHEN
        result = self.state._combine_all_channels(
            processed_categorical, processed_numerical, spatial_shape
        )

        # THEN
        self.assertEqual(result.shape, (5, self.height, self.width))
        np.testing.assert_array_equal(result[:2], processed_categorical)
        np.testing.assert_array_equal(result[2:], processed_numerical)

    def test_combine_all_channels_empty(self):
        # GIVEN
        processed_categorical = None
        processed_numerical = None
        spatial_shape = (self.height, self.width)

        # WHEN
        result = self.state._combine_all_channels(
            processed_categorical, processed_numerical, spatial_shape
        )

        # THEN
        self.assertEqual(result.shape, (0, self.height, self.width))


class TestSpatialFeatures(unittest.TestCase):
    """Test spatial feature processing (screen and minimap)"""
    
    def setUp(self):
        self.state = DeepmindState()
    
    @patch('urnai.sc2.states.deepmind_state.DeepmindState._process_spatial_features')
    def test_process_screen(self, mock_process_spatial_features):
        # GIVEN
        obs = {'feature_screen': np.array([[[1, 2], [3, 4]]])}
        expected_result = np.array([[[5, 6], [7, 8]]])
        mock_process_spatial_features.return_value = expected_result

        # WHEN
        result = self.state.process_screen(obs)

        # THEN
        mock_process_spatial_features.assert_called_once_with(
            obs['feature_screen'], features.SCREEN_FEATURES
        )
        np.testing.assert_array_equal(result, expected_result)

    @patch('urnai.sc2.states.deepmind_state.DeepmindState._process_spatial_features')
    def test_process_minimap(self, mock_process_spatial_features):
        # GIVEN
        obs = {'feature_minimap': np.array([[[1, 2], [3, 4]]])}
        expected_result = np.array([[[5, 6], [7, 8]]])
        mock_process_spatial_features.return_value = expected_result

        # WHEN
        result = self.state.process_minimap(obs)

        # THEN
        mock_process_spatial_features.assert_called_once_with(
            obs['feature_minimap'], features.MINIMAP_FEATURES
        )
        np.testing.assert_array_equal(result, expected_result)

    def test_process_spatial_features(self):
        # GIVEN
        height = 4
        width = 4
        feature_specs = [
            Mock(type=features.FeatureType.CATEGORICAL, scale=3),
            Mock(type=features.FeatureType.SCALAR, scale=10),
            Mock(type=features.FeatureType.CATEGORICAL, scale=2),
            Mock(type=features.FeatureType.SCALAR, scale=5)
        ]
        raw_features = np.arange(
            len(feature_specs) * height * width, dtype=np.float32
            ).reshape((len(feature_specs), height, width))
        
        expected_categorical_items = [
            (raw_features[0], feature_specs[0]),
            (raw_features[2], feature_specs[2])
        ]
        expected_numerical_items = [
            (raw_features[1], feature_specs[1]),
            (raw_features[3], feature_specs[3])
        ]
        
        processed_categorical = np.full(
            (self.state.categorical_out_conv_channels, height, width), 
            1.0, 
            dtype=np.float32
        )
        processed_numerical = np.full((2, height, width), 2.0, dtype=np.float32)

        with patch.object(
            self.state, '_separate_raw_channels_by_type'
        ) as mock_separate, \
             patch.object(self.state, '_process_categorical_channels') as mock_cat, \
             patch.object(self.state, '_process_numerical_channels') as mock_num, \
             patch.object(self.state, '_combine_all_channels') as mock_combine:
            
            mock_separate.return_value = (
                expected_categorical_items, expected_numerical_items
            )
            mock_cat.return_value = processed_categorical
            mock_num.return_value = processed_numerical
            mock_combine.return_value = np.concatenate(
                [processed_categorical, processed_numerical], axis=0
            )

            # WHEN
            result = self.state._process_spatial_features(raw_features, feature_specs)

            # THEN
            mock_separate.assert_called_once_with(raw_features, feature_specs)
            mock_cat.assert_called_once_with(expected_categorical_items)
            mock_num.assert_called_once_with(expected_numerical_items)
            mock_combine.assert_called_once_with(
                processed_categorical, processed_numerical, (height, width)
            )
            self.assertIsNotNone(result)


class TestNonSpatialFeatures(unittest.TestCase):
    """Test non-spatial feature processing"""
    
    def setUp(self):
        self.state = DeepmindState()
    
    def test_process_non_spatial(self):
        # GIVEN
        player_data = np.array([100, 200, 50], dtype=np.int32)
        obs = {'player': player_data}

        # WHEN
        result = self.state.process_non_spatial(obs)

        # THEN
        expected = np.log1p(player_data.astype(np.float32))
        np.testing.assert_array_equal(result, expected)


class TestStateManagement(unittest.TestCase):
    """Test state management and dimension handling"""
    
    def setUp(self):
        self.state = DeepmindState()
    
    def test_update_and_dimension(self):
        # GIVEN
        screen_array = np.ones((5, 64, 64), dtype=np.float32)
        minimap_array = np.ones((7, 64, 64), dtype=np.float32)
        player_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        with patch.object(self.state, 'process_screen') as mock_screen, \
             patch.object(self.state, 'process_minimap') as mock_minimap, \
             patch.object(self.state, 'process_non_spatial') as mock_non_spatial:
            
            mock_screen.return_value = screen_array
            mock_minimap.return_value = minimap_array
            mock_non_spatial.return_value = player_array

            obs = {'fake': 'observation'}

            # WHEN
            self.state.update(obs)

            # THEN
            self.assertIsNotNone(self.state._state)
            self.assertEqual(len(self.state._state), 3)
            np.testing.assert_array_equal(self.state._state[0], screen_array)
            np.testing.assert_array_equal(self.state._state[1], minimap_array)
            np.testing.assert_array_equal(self.state._state[2], player_array)

            dims = self.state.dimension
            self.assertEqual(dims[0], screen_array.shape)
            self.assertEqual(dims[1], minimap_array.shape)
            self.assertEqual(dims[2], player_array.shape)

    def test_dimension_no_state(self):
        # GIVEN
        # State is None by default

        # WHEN
        dims = self.state.dimension

        # THEN
        self.assertIsNone(dims)

    def test_dimension_invalid_state(self):
        # GIVEN
        self.state._state = [1, 2]  # Invalid state with wrong length

        # WHEN
        dims = self.state.dimension

        # THEN
        self.assertIsNone(dims)

    def test_state_property_initial_none(self):
        # GIVEN
        # State is None by default after initialization

        # WHEN
        result = self.state.state

        # THEN
        self.assertIsNone(result)

    def test_state_property_after_update(self):
        # GIVEN
        expected_state = [
            np.ones((5, 64, 64), dtype=np.float32),  # screen
            np.ones((7, 64, 64), dtype=np.float32),  # minimap
            np.array([1.0, 2.0, 3.0], dtype=np.float32)  # player
        ]
        self.state._state = expected_state

        # WHEN
        result = self.state.state

        # THEN
        self.assertIs(result, expected_state)  # Should return the same object reference
        self.assertEqual(len(result), 3)
        np.testing.assert_array_equal(result[0], expected_state[0])
        np.testing.assert_array_equal(result[1], expected_state[1])
        np.testing.assert_array_equal(result[2], expected_state[2])
