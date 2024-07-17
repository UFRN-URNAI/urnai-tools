import unittest

from pysc2.lib.named_array import NamedDict

from urnai.sc2.states.starcraft2_state import StarCraft2State


class TestStarCraft2State(unittest.TestCase):


    def test_starcraft2_state_no_raw(self):
        # GIVEN
        state = StarCraft2State(use_raw_units=False)
        obs = NamedDict({
            'player': NamedDict({
                'minerals': 100,
                'vespene': 100,
                'food_cap': 200,
                'food_used': 100,
                'food_army': 50,
                'food_workers': 50,
                'army_count': 20,
                'idle_worker_count': 10,
            })
        })
        # WHEN
        state.update(obs)
        # THEN
        assert state.dimension == 9
        assert len(state.state[0]) == 9

    def test_starcraft2_state_raw(self):
        # GIVEN
        state = StarCraft2State(grid_size=5, use_raw_units=True)
        obs = NamedDict({
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
        state.update(obs)
        # THEN
        assert state.dimension == 9 + ((5 * 5) * 2)
        assert len(state.state[0]) == 9 + ((5 * 5) * 2)
