import unittest

import numpy as np
from pysc2.lib import features
from pysc2.lib.named_array import NamedDict

from urnai.sc2.rewards.collectables import CollectablesReward

MAXIMUM_NUMBER_OF_MINERAL_SHARDS = 2
EXAMPLE_OBSERVATION = NamedDict({
    'player': NamedDict({
        'minerals': 100,
        'vespene': 100,
        'food_cap': 200,
        'food_used': 100,
        'food_army': 50,
        'food_workers': 50,
        'army_count': 20,
        'idle_worker_count': 10,
    }),
    'raw_units': [
        NamedDict({
            'unit_type': 341,
            'alliance': features.PlayerRelative.NEUTRAL,
            'build_progress': 100,
            'x': 1,
            'y': 0,
            'tag': 0,
        }),
        NamedDict({
            'unit_type': 341,
            'alliance': features.PlayerRelative.NEUTRAL,
            'build_progress': 100,
            'x': 2,
            'y': 3,
            'tag': 0,
        }),
    ],
    'feature_minimap': [[[0 for w in range(4)] for h in range(4)]]
})

class TestCollectablesReward(unittest.TestCase):

    def test_get(self):
        # GIVEN
        reward = CollectablesReward()
        obs, default_reward, terminated, truncated \
            = EXAMPLE_OBSERVATION, None, False, False
        # WHEN
        get_return = reward.get(obs, default_reward, terminated, truncated)
        # THEN
        assert get_return == 0
        self.assertEqual(reward.previous_state, obs)

        # GIVEN
        reward = CollectablesReward()
        reward.old_collectable_counter = MAXIMUM_NUMBER_OF_MINERAL_SHARDS
        reward.previous_state = EXAMPLE_OBSERVATION
        obs, default_reward, terminated, truncated = \
            EXAMPLE_OBSERVATION, None, False, False
        # WHEN
        get_return = reward.get(obs, default_reward, terminated, truncated)
        # THEN
        assert get_return == -1
        self.assertEqual(reward.previous_state, obs)

        # GIVEN
        reward = CollectablesReward()
        reward.old_collectable_counter = MAXIMUM_NUMBER_OF_MINERAL_SHARDS - 1
        reward.previous_state = EXAMPLE_OBSERVATION
        obs, default_reward, terminated, truncated = \
            EXAMPLE_OBSERVATION, None, False, False
        # WHEN
        get_return = reward.get(obs, default_reward, terminated, truncated)
        # THEN
        assert get_return == 10
        self.assertEqual(reward.previous_state, obs)

        # GIVEN
        reward = CollectablesReward()
        reward.old_collectable_counter = MAXIMUM_NUMBER_OF_MINERAL_SHARDS - 1
        reward.previous_state = EXAMPLE_OBSERVATION
        obs, default_reward, terminated, truncated = \
            EXAMPLE_OBSERVATION, None, True, False
        # WHEN
        get_return = reward.get(obs, default_reward, terminated, truncated)
        # THEN
        assert get_return == 500
        self.assertEqual(reward.previous_state, obs)
    
    def test_reset(self):
        # GIVEN
        reward = CollectablesReward()
        initial_old_collectable_counter = reward.old_collectable_counter
        # WHEN
        reward.reset()
        # THEN
        assert reward.previous_state is None
        assert reward.old_collectable_counter == initial_old_collectable_counter
        assert reward.score == 0

    def test_filter_non_mineral_shard_units(self):
        # GIVEN
        reward = CollectablesReward()
        reward.old_collectable_counter = MAXIMUM_NUMBER_OF_MINERAL_SHARDS
        obs = EXAMPLE_OBSERVATION
        # WHEN
        filter_return = reward.filter_non_mineral_shard_units(obs)
        filter_expected_return = np.zeros((4, 4))
        filter_expected_return[0][1] = 1
        filter_expected_return[3][2] = 1
        # THEN
        assert np.array_equal(filter_return,  filter_expected_return)