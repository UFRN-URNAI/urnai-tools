import unittest

from pysc2.lib import units
from pysc2.lib.named_array import NamedDict

import urnai.sc2.actions.library_sc2 as libsc2
from urnai.sc2.actions.sc2_actions import raw_functions_classes as sc2_actions

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
            'unit_type': units.Terran.Marine,
            'alliance': 1,
            'build_progress': 100,
            'x': 1,
            'y': 1,
            'tag': 0,
        }),
        NamedDict({
            'unit_type': units.Terran.Marine,
            'alliance': 1,
            'build_progress': 100,
            'x': 2,
            'y': 2,
            'tag': 1,
        }),
    ]
})

class TestLibrarySc2(unittest.TestCase):

    def test_move_direction(self):
        # GIVEN
        obs = EXAMPLE_OBSERVATION
        direction = {"x" : 1, "y" : 0}

        # WHEN
        actions = libsc2.move_direction(obs, direction)

        # THEN
        assert actions == [
            sc2_actions["Move_pt"].run('now', 0, [2, 1]),
            sc2_actions["Move_pt"].run('now', 1, [2, 1])
        ]

    def test_move_left(self):
        # GIVEN
        obs = EXAMPLE_OBSERVATION
        
        # WHEN
        actions = libsc2.move_left(obs)

        # THEN
        assert actions == [
            sc2_actions["Move_pt"].run('now', 0, [-1, 1]),
            sc2_actions["Move_pt"].run('now', 1, [-1, 1])
        ]
    
    def test_move_right(self):
        # GIVEN
        obs = EXAMPLE_OBSERVATION
        
        # WHEN
        actions = libsc2.move_right(obs)

        # THEN
        assert actions == [
            sc2_actions["Move_pt"].run('now', 0, [3, 1]),
            sc2_actions["Move_pt"].run('now', 1, [3, 1])
        ]
    
    def test_move_up(self):
        # GIVEN
        obs = EXAMPLE_OBSERVATION
        
        # WHEN
        actions = libsc2.move_up(obs)

        # THEN
        assert actions == [
            sc2_actions["Move_pt"].run('now', 0, [1, -1]),
            sc2_actions["Move_pt"].run('now', 1, [1, -1])
        ]

    def test_move_down(self):
        # GIVEN
        obs = EXAMPLE_OBSERVATION
        
        # WHEN
        actions = libsc2.move_down(obs)

        # THEN
        assert actions == [
            sc2_actions["Move_pt"].run('now', 0, [1, 3]),
            sc2_actions["Move_pt"].run('now', 1, [1, 3])
        ]

