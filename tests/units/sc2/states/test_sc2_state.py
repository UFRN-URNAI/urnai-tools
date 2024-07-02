import unittest

from pysc2.lib.named_array import NamedDict

from urnai.sc2.states.protoss_state import ProtossState
from urnai.sc2.states.terran_state import TerranState
from urnai.sc2.states.zerg_state import ZergState


class TestSC2State(unittest.TestCase):


    def test_terran_state_no_raw(self):
        # GIVEN
        state = TerranState(use_raw_units=False)
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
        print(state.state)
        assert state.dimension == 9
        assert len(state.state[0]) == 9

    def test_terran_state_raw(self):
        # GIVEN
        state = TerranState(grid_size=4, use_raw_units=True)
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
        assert state.dimension == 22 + ((4 * 4) * 2)
        assert len(state.state[0]) == 22 + ((4 * 4) * 2)

    def test_protoss_state_no_raw(self):
        # GIVEN
        state = ProtossState(use_raw_units=False)
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
        print(state.state)
        assert state.dimension == 9
        assert len(state.state[0]) == 9

    def test_protoss_state_raw(self):
        # GIVEN
        state = ProtossState(grid_size=5, use_raw_units=True)
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
        assert state.dimension == 22 + ((5 * 5) * 2)
        assert len(state.state[0]) == 22 + ((5 * 5) * 2)
    
    def test_zerg_state_no_raw(self):
        # GIVEN
        state = ZergState(use_raw_units=False)
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
        print(state.state)
        assert state.dimension == 9
        assert len(state.state[0]) == 9

    def test_zerg_state_raw(self):
        # GIVEN
        state = ZergState(grid_size=6, use_raw_units=True)
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
        assert state.dimension == 22 + ((6 * 6) * 2)
        assert len(state.state[0]) == 22 + ((6 * 6) * 2)
