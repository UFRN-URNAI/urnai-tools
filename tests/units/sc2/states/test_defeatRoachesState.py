import numpy as np
import unittest

from urnai.sc2.states.defeatRoaches import DefeatRoachesState
from urnai.sc2.states.experiments import StateType

from pysc2.lib import units
from pysc2.lib.named_array import NamedDict


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

EXAMPLE_MAP = np.zeros((64, 64, 3))
points_to_color = [(10, 5), (24, 27), (48, 50)]
for point in points_to_color:
    EXAMPLE_MAP[point[1]][point[0]] = (255, 0, 0)

NORMAL_SIZED_MAP_OBSERVATION = NamedDict({
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
    'feature_minimap': [EXAMPLE_MAP],
})

class TestDefeatRoachesState(unittest.TestCase):

    def test_build_basic_map(self):
        # GIVEN
        state = DefeatRoachesState()
        obs = EXAMPLE_OBSERVATION
        expected_map = np.zeros(
            (obs.feature_minimap[0].shape[0],
            obs.feature_minimap[0].shape[1], 3), dtype=np.uint8)
        
        expected_map[1][1] = (255, 0, 0)
        expected_map[2][2] = (255, 0, 0)
        expected_map[3][3] = (0, 255, 0)
        expected_map[0][0] = (0, 255, 0)

        # WHEN
        returned_map = state.build_basic_map(obs)
        
        # THEN
        assert (returned_map == expected_map).all()

    def test_reduce_map(self):
        # GIVEN
        state = DefeatRoachesState(trim_map=False)
        obs = NORMAL_SIZED_MAP_OBSERVATION
        expected_shape = obs.feature_minimap[0].shape

        # WHEN
        reduced_map = state.reduce_map(obs.feature_minimap[0])

        # THEN
        assert reduced_map.shape == expected_shape
        assert np.array_equal(reduced_map, obs.feature_minimap[0])

    def test_reduce_map_trim(self):
        # GIVEN
        state = DefeatRoachesState(trim_map=True)
        obs = NORMAL_SIZED_MAP_OBSERVATION
        expected_shape = (17, 23, 3)

        # WHEN
        reduced_map = state.reduce_map(obs.feature_minimap[0])

        # THEN
        assert reduced_map.shape == expected_shape
        for y in range(expected_shape[0]):
            for x in range(expected_shape[1]):
                if not np.array_equal(reduced_map[y][x], obs.feature_minimap[0][20 + y][22 + x]):
                    assert False

    def test_reduce_map_trim_and_reduction(self):
        # GIVEN
        state = DefeatRoachesState(trim_map=True)
        state.map_reduction_factor = 4
        obs = NORMAL_SIZED_MAP_OBSERVATION
        expected_shape = (4, 5, 3)
        expected_map = [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[15.9375, 15.9375, 15.9375], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ]

        # WHEN
        reduced_map = state.reduce_map(obs.feature_minimap[0])

        # THEN
        assert reduced_map.shape == expected_shape
        np.array_equal(reduced_map, expected_map)

    def test_reduce_reduction(self):
        # GIVEN
        state = DefeatRoachesState(trim_map=False)
        state.map_reduction_factor = 16
        obs = NORMAL_SIZED_MAP_OBSERVATION
        expected_shape = (4, 4, 3)
        expected_map = np.array([
            [[0.99609375, 0.99609375, 0.99609375], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0.99609375, 0.99609375, 0.99609375], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0.99609375, 0.99609375, 0.99609375]]
        ])

        # WHEN
        reduced_map = state.reduce_map(obs.feature_minimap[0])

        # THEN
        assert reduced_map.shape == expected_shape
        assert np.array_equal(reduced_map, expected_map)
