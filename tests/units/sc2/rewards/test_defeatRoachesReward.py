import unittest

import numpy as np
from pysc2.lib import units
from pysc2.lib.named_array import NamedDict

from urnai.sc2.rewards.defeatRoaches import DefeatRoachesReward

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
    'feature_minimap': [np.zeros((4, 4))],
    'raw_units': [
        NamedDict({
            'unit_type': units.Terran.Marine,
            'alliance': 1,
            'build_progress': 100,
            'x': 1,
            'y': 1,
            'tag': 0,
            'owner': 1,
        }),
        NamedDict({
            'unit_type': units.Terran.Marine,
            'alliance': 1,
            'build_progress': 100,
            'x': 2,
            'y': 2,
            'tag': 1,
            'owner': 1,
        }),
        NamedDict({
            'unit_type': units.Terran.Marine,
            'alliance': 1,
            'build_progress': 100,
            'x': 6,
            'y': 6,
            'tag': 4,
            'owner': 1,
        }),
        NamedDict({
            'unit_type': units.Zerg.Roach,
            'alliance': 4,
            'build_progress': 100,
            'x': 3,
            'y': 3,
            'tag': 2,
            'owner': 4,
        }),
        NamedDict({
            'unit_type': units.Zerg.Roach,
            'alliance': 4,
            'build_progress': 100,
            'x': 0,
            'y': 0,
            'tag': 3,
            'owner': 4,
        }),
    ]
})

EXAMPLE_OBSERVATION_FUTURE = NamedDict({
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
    'feature_minimap': [np.zeros((4, 4))],
    'raw_units': [
        NamedDict({
            'unit_type': units.Terran.Marine,
            'alliance': 1,
            'build_progress': 100,
            'x': 1,
            'y': 1,
            'tag': 0,
            'owner': 1,
        }),
        NamedDict({
            'unit_type': units.Terran.Marine,
            'alliance': 1,
            'build_progress': 100,
            'x': 2,
            'y': 2,
            'tag': 1,
            'owner': 1,
        }),
        NamedDict({
            'unit_type': units.Terran.Marine,
            'alliance': 1,
            'build_progress': 100,
            'x': 6,
            'y': 6,
            'tag': 4,
            'owner': 1,
        }),
        NamedDict({
            'unit_type': units.Zerg.Roach,
            'alliance': 4,
            'build_progress': 100,
            'x': 0,
            'y': 0,
            'tag': 3,
            'owner': 4,
        }),
    ]
})


class TestDefeatRoachesReward(unittest.TestCase):

    def test_get(self):
        # GIVEN
        reward = DefeatRoachesReward()
        obs = EXAMPLE_OBSERVATION

        # WHEN
        returned_reward = reward.get(obs, None, False, False)

        # THEN
        assert returned_reward == 0

    def test_get_with_prev_state(self):
        # GIVEN
        reward = DefeatRoachesReward()
        obs = EXAMPLE_OBSERVATION
        reward.previous_state = obs

        # WHEN
        returned_reward = reward.get(obs, None, False, False)

        # THEN
        assert returned_reward == 0

    def test_get_truncated_or_terminated(self):
        # GIVEN
        reward = DefeatRoachesReward()
        obs = EXAMPLE_OBSERVATION
        reward.previous_state = obs

        # WHEN
        returned_reward = reward.get(obs, None, True, False)

        # THEN
        assert returned_reward == 2500

    def test_get_with_future(self):
        # GIVEN
        reward = DefeatRoachesReward()
        obs = EXAMPLE_OBSERVATION_FUTURE
        reward.previous_state = EXAMPLE_OBSERVATION

        # WHEN
        returned_reward = reward.get(obs, None, False, False)

        # THEN
        assert returned_reward == 1000

    def test_get_with_future_terminated_or_truncated(self):
        # GIVEN
        reward = DefeatRoachesReward()
        obs = EXAMPLE_OBSERVATION_FUTURE
        reward.previous_state = EXAMPLE_OBSERVATION

        # WHEN
        returned_reward = reward.get(obs, None, True, False)

        # THEN
        assert returned_reward == 6000

    def test_get_roach_amount(self):
        # GIVEN
        reward = DefeatRoachesReward()
        obs = EXAMPLE_OBSERVATION

        # WHEN
        n_roaches = reward.get_roach_amount(obs)

        # THEN
        assert n_roaches == 2

    def test_get_marine_amount(self):
        # GIVEN
        reward = DefeatRoachesReward()
        obs = EXAMPLE_OBSERVATION

        # WHEN
        n_marines = reward.get_marine_amount(obs)

        # THEN
        assert n_marines == 3