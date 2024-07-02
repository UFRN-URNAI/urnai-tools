import unittest

from pysc2.lib.named_array import NamedDict

from urnai.sc2.states.states_utils import (
    append_player_and_enemy_grids,
    get_my_raw_units_amount,
    get_raw_units_by_type,
)


class TestAuxSC2State(unittest.TestCase):

    def test_append_player_and_enemy_grids(self):
        # GIVEN
        obs = NamedDict({
            'raw_units': [
                NamedDict({
                    'unit_type': 1,
                    'alliance': 1,
                    'build_progress': 100,
                    'x': 1,
                    'y': 1,
                }),
                NamedDict({
                    'unit_type': 2,
                    'alliance': 1,
                    'build_progress': 100,
                    'x': 2,
                    'y': 2,
                }),
                NamedDict({
                    'unit_type': 2,
                    'alliance': 4,
                    'build_progress': 100,
                    'x': 63,
                    'y': 63,
                }),
            ]
        })
        new_state = []
        # WHEN
        new_state = append_player_and_enemy_grids(obs, new_state, 3, 64)
        # THEN
        print(new_state)
        assert len(new_state) == (18)
        assert new_state[8] == 0.005
        assert new_state[9] == 0.01
    
    def test_get_my_raw_units_amount(self):
        # GIVEN
        obs = NamedDict({
            'raw_units': [
                NamedDict({
                    'unit_type': 1,
                    'alliance': 1,
                    'build_progress': 100,
                    'x': 1,
                    'y': 1,
                }),
                NamedDict({
                    'unit_type': 2,
                    'alliance': 1,
                    'build_progress': 100,
                    'x': 2,
                    'y': 2,
                }),
            ]
        })
        # WHEN
        amount = get_my_raw_units_amount(obs, 1)
        # THEN
        assert amount == 1
    
    def test_get_my_raw_units_amount_no_units(self):
        # GIVEN
        obs = NamedDict({
            'raw_units': []
        })
        # WHEN
        amount = get_my_raw_units_amount(obs, 1)
        # THEN
        assert amount == 0
    
    def test_get_my_raw_units_amount_no_units_of_type(self):
        # GIVEN
        obs = NamedDict({
            'raw_units': [
                NamedDict({
                    'unit_type': 2,
                    'alliance': 1,
                    'build_progress': 100,
                    'x': 1,
                    'y': 1,
                }),
                NamedDict({
                    'unit_type': 2,
                    'alliance': 1,
                    'build_progress': 100,
                    'x': 2,
                    'y': 2,
                }),
            ]
        })
        # WHEN
        amount = get_my_raw_units_amount(obs, 1)
        # THEN
        assert amount == 0
    
    def test_get_my_raw_units_amount_no_units_of_alliance(self):
        # GIVEN
        obs = NamedDict({
            'raw_units': [
                NamedDict({
                    'unit_type': 1,
                    'alliance': 2,
                    'build_progress': 100,
                    'x': 1,
                    'y': 1,
                }),
                NamedDict({
                    'unit_type': 1,
                    'alliance': 2,
                    'build_progress': 100,
                    'x': 2,
                    'y': 2,
                }),
            ]
        })
        # WHEN
        amount = get_my_raw_units_amount(obs, 1)
        # THEN
        assert amount == 0
    
    def test_get_my_raw_units_amount_no_units_of_build_progress(self):
        # GIVEN
        obs = NamedDict({
            'raw_units': [
                NamedDict({
                    'unit_type': 1,
                    'alliance': 1,
                    'build_progress': 50,
                    'x': 1,
                    'y': 1,
                }),
                NamedDict({
                    'unit_type': 1,
                    'alliance': 1,
                    'build_progress': 50,
                    'x': 2,
                    'y': 2,
                }),
            ]
        })
        # WHEN
        amount = get_my_raw_units_amount(obs, 1)
        # THEN
        assert amount == 0
    
    def test_get_raw_units_by_type(self):
        # GIVEN
        obs = NamedDict({
            'raw_units': [
                NamedDict({
                    'unit_type': 1,
                    'alliance': 1,
                    'build_progress': 100,
                    'x': 1,
                    'y': 1,
                }),
                NamedDict({
                    'unit_type': 2,
                    'alliance': 1,
                    'build_progress': 100,
                    'x': 2,
                    'y': 2,
                }),
            ]
        })
        # WHEN
        units = get_raw_units_by_type(obs, 1)
        # THEN
        assert len(units) == 1
    
    def test_get_raw_units_by_type_no_units(self):
        # GIVEN
        obs = NamedDict({
            'raw_units': []
        })
        # WHEN
        units = get_raw_units_by_type(obs, 1)
        # THEN
        assert len(units) == 0
    
    def test_get_raw_units_by_type_no_units_of_type(self):
        # GIVEN
        obs = NamedDict({
            'raw_units': [
                NamedDict({
                    'unit_type': 2,
                    'alliance': 1,
                    'build_progress': 100,
                    'x': 1,
                    'y': 1,
                }),
                NamedDict({
                    'unit_type': 2,
                    'alliance': 1,
                    'build_progress': 100,
                    'x': 2,
                    'y': 2,
                }),
            ]
        })
        # WHEN
        units = get_raw_units_by_type(obs, 1)
        # THEN
        assert len(units) == 0