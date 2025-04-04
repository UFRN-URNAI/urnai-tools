import unittest

from pysc2.lib import units
from pysc2.lib.named_array import NamedDict

from urnai.sc2.actions.defeatRoaches import DefeatRoachesActionSpace
from urnai.sc2.actions.library_sc2 import move_down, move_left, move_right, move_up
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
            'x': 10,
            'y': 10,
            'tag': 2,
            'owner': 4,
        }),
        NamedDict({
            'unit_type': units.Zerg.Roach,
            'alliance': 4,
            'build_progress': 100,
            'x': 12,
            'y': 12,
            'tag': 3,
            'owner': 4,
        }),
    ]
})

EMPTY_OBSERVATION = NamedDict({
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
    'raw_units': []
})

EXAMPLE_ARMY = [
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
    })
]

class TestDefeatRoachesActionSpace(unittest.TestCase):

    def test_solve_action(self):
        # GIVEN
        actionSpace = DefeatRoachesActionSpace()
        action_ids = [8, 9, 10, 11, 0, 5]
        obs = EXAMPLE_OBSERVATION
        action_calls = [move_left(obs), move_right(obs), move_up(obs),
                        move_down(obs), actionSpace.attack(obs),
                        actionSpace.runaway(obs)]
        expected_actions = []
        for action_call in action_calls:
            expected_actions += action_call
        # WHEN
        for action_idx in action_ids:
            actionSpace.solve_action(action_idx, obs)
        
        # THEN
        assert actionSpace.pending_actions == expected_actions

        # GIVEN
        actionSpace = DefeatRoachesActionSpace()
        action_idx = None
        obs = EXAMPLE_OBSERVATION
        # WHEN
        actionSpace.solve_action(action_idx, obs)
        # THEN
        assert actionSpace.pending_actions == []

    def test_get_nearest_enemy_unit_inside_radius(self):
        # GIVEN
        actionSpace = DefeatRoachesActionSpace()
        obs = EXAMPLE_OBSERVATION

        # WHEN
        nearest_enemy_unit = actionSpace.get_nearest_enemy_unit_inside_radius(
            0, 0, obs, 20
        )

        # THEN
        assert nearest_enemy_unit.x == 10

        # GIVEN
        actionSpace = DefeatRoachesActionSpace()
        obs = EMPTY_OBSERVATION

        # WHEN
        nearest_enemy_unit = actionSpace.get_nearest_enemy_unit_inside_radius(
            0, 0, obs, 20
        )

        # THEN
        assert nearest_enemy_unit is None

    def test_get_army_avg(self):
        # GIVEN
        actionSpace = DefeatRoachesActionSpace()
        army = EXAMPLE_ARMY

        # WHEN
        army_avg = actionSpace.get_army_avg(army)

        # THEN
        assert army_avg == (1, 1)

    def test_attack_nearest_inside_radius(self):
        # GIVEN
        actionSpace = DefeatRoachesActionSpace()
        obs = EXAMPLE_OBSERVATION

        # WHEN
        returned_actions = actionSpace.attack_nearest_inside_radius(obs, 20)

        # THEN
        assert returned_actions == [
            sc2_actions['Attack_unit'].run('now', 0, 2),
            sc2_actions['Attack_unit'].run('now', 1, 2)
        ]

        # GIVEN
        actionSpace = DefeatRoachesActionSpace()
        obs = EMPTY_OBSERVATION

        # THEN
        assert actionSpace.attack_nearest_inside_radius(obs, 20) == []

    def test_attack(self):
        # GIVEN
        actionSpace = DefeatRoachesActionSpace()
        obs = EXAMPLE_OBSERVATION

        # WHEN
        returned_actions = actionSpace.attack(obs)

        # THEN
        assert returned_actions == [
            sc2_actions['Attack_unit'].run('now', 0, 2),
            sc2_actions['Attack_unit'].run('now', 1, 2)
        ]

        # GIVEN
        actionSpace = DefeatRoachesActionSpace()
        obs = EMPTY_OBSERVATION

        # WHEN
        returned_actions = actionSpace.attack(obs)

        # THEN
        assert returned_actions == []

    def test_runaway(self):
        # GIVEN
        actionSpace = DefeatRoachesActionSpace()
        obs = EXAMPLE_OBSERVATION

        # WHEN
        returned_actions = actionSpace.runaway(obs)

        # THEN
        assert returned_actions == [
            sc2_actions['Move_pt'].run('now', 0, [-1, -1]),
            sc2_actions['Move_pt'].run('now', 1, [-1, -1])
        ]

        # GIVEN
        actionSpace = DefeatRoachesActionSpace()
        obs = EMPTY_OBSERVATION

        # WHEN
        returned_actions = actionSpace.runaway(obs)

        # THEN
        assert returned_actions == []