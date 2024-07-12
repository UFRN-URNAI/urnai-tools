import unittest

from pysc2.lib.named_array import NamedDict

from urnai.sc2.states.utils import (
    append_player_and_enemy_grids,
   create_raw_units_amount_dict,
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
        assert len(new_state) == (18)
        assert new_state[8] == 0.005
        assert new_state[9] == 0.01
    
    def test_create_raw_units_amount_dict(self):
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
        # WHEN
        dict = create_raw_units_amount_dict(obs, 1)
        # THEN
        assert len(dict) == 2
        assert dict[1] == 1
        assert dict[2] == 1
    
    def test_create_raw_units_amount_dict_alliance(self):
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
        # WHEN
        dict = create_raw_units_amount_dict(obs, 4)
        # THEN
        assert len(dict) == 1
        assert dict[1] == 0
        assert dict[2] == 1
    
    def test_create_raw_units_amount_dict_no_units(self):
        # GIVEN
        obs = NamedDict({
            'raw_units': []
        })
        # WHEN
        dict = create_raw_units_amount_dict(obs)
        # THEN
        assert len(dict) == 0
        assert dict == {}
    
    def test_create_raw_units_amount_dict_no_units_alliance(self):
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
        # WHEN
        dict = create_raw_units_amount_dict(obs, 3)
        # THEN
        assert len(dict) == 0
        assert dict == {}