import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from pysc2.lib import units as sc2units

from urnai.sc2.states.buildmarines import (
    MAXIMUM_NUMBER_OF_BARRACKS,
    MAXIMUM_NUMBER_OF_MARINES,
    MAXIMUM_NUMBER_OF_SUPPLY_DEPOT,
    STATE_MAXIMUM_MINERALS,
    BuildMarinesState,
)


class TestBuildMarinesState(unittest.TestCase):

    def setUp(self):
        self.state_builder = BuildMarinesState()

    def test_initialization(self):
        # GIVEN / WHEN : setUp()
        # THEN
        self.assertEqual(self.state_builder.non_spatial_state, [0, 0, 0, 0])
        self.assertIsNone(self.state_builder.state)
        
        self.assertEqual(self.state_builder.non_spatial_maximums[0],
                         STATE_MAXIMUM_MINERALS)
        self.assertEqual(self.state_builder.non_spatial_maximums[1],
                         MAXIMUM_NUMBER_OF_SUPPLY_DEPOT)
        self.assertEqual(self.state_builder.non_spatial_maximums[2],
                         MAXIMUM_NUMBER_OF_BARRACKS)
        self.assertEqual(self.state_builder.non_spatial_maximums[3], 
                         MAXIMUM_NUMBER_OF_MARINES)

    def test_reset(self):
        # GIVEN
        self.state_builder._state = np.array([1, 2, 3, 4])
        self.state_builder.non_spatial_state = [1, 1, 1, 1]
        # WHEN
        self.state_builder.reset()
        # THEN
        self.assertIsNone(self.state_builder.state)
        self.assertEqual(self.state_builder.non_spatial_state, [0, 0, 0, 0])

    def test_normalize_value(self):
        # GIVEN
        test_cases = [
            ("middle value", 50, 100, 0, 0.5),
            ("min value", 0, 100, 0, 0.0),
            ("max value", 100, 100, 0, 1.0),
            ("min not equal to zero", 75, 100, 50, 0.5),
        ]
        for description, value, max_val, min_val, expected in test_cases:
            with self.subTest(description=description):
                # WHEN
                result = self.state_builder.normalize_value(value, max_val, min_val)
                
                # THEN
                self.assertAlmostEqual(result, expected)

    def test_normalize_non_spatial_list(self):
        # GIVEN
        initial_state_values = [500, 2, 3, 20]
        maximum_values = [
            STATE_MAXIMUM_MINERALS,
            MAXIMUM_NUMBER_OF_SUPPLY_DEPOT,
            MAXIMUM_NUMBER_OF_BARRACKS,
            MAXIMUM_NUMBER_OF_MARINES
        ]
        self.state_builder.non_spatial_state = initial_state_values.copy()
        self.state_builder.non_spatial_maximums = maximum_values
        # WHEN
        self.state_builder.normalize_non_spatial_list()
        # THEN
        expected_normalized_state = np.array(initial_state_values) / \
                                            np.array(maximum_values)
        np.testing.assert_array_almost_equal(
            self.state_builder.non_spatial_state, 
            expected_normalized_state
        )

    @patch('urnai.sc2.actions.sc2_actions_aux.get_my_units_amount')
    def test_update_and_build_non_spatial_state(self, mock_get_units_amount):
        # GIVEN
        mock_obs = MagicMock()
        mock_obs.player.minerals = 500
        mock_get_units_amount.side_effect = [
            2,  # SupplyDepot
            3,  # Barracks
            20  # Marine
        ]
        # WHEN
        final_state = self.state_builder.update(mock_obs)
        # THEN
        expected_minerals = 500 / STATE_MAXIMUM_MINERALS
        expected_supply_depots = 2 / MAXIMUM_NUMBER_OF_SUPPLY_DEPOT
        expected_barracks = 3 / MAXIMUM_NUMBER_OF_BARRACKS
        expected_marines = 20 / MAXIMUM_NUMBER_OF_MARINES
        expected_state = np.array([
            expected_minerals,
            expected_supply_depots,
            expected_barracks,
            expected_marines
        ])
        self.assertIsInstance(final_state, np.ndarray)
        np.testing.assert_array_almost_equal(final_state, expected_state)
        np.testing.assert_array_equal(self.state_builder.state, final_state)
        self.assertEqual(self.state_builder.dimension, 4)
        self.assertEqual(mock_get_units_amount.call_count, 3)
        calls = mock_get_units_amount.call_args_list
        self.assertEqual(calls[0].args[1], sc2units.Terran.SupplyDepot)
        self.assertEqual(calls[1].args[1], sc2units.Terran.Barracks)
        self.assertEqual(calls[2].args[1], sc2units.Terran.Marine)