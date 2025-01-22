import unittest

from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib.named_array import NamedDict

from urnai.sc2.actions.collectables import CollectablesActionSpace
from urnai.sc2.actions.sc2_action import SC2Action

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
            'unit_type': 48,
            'alliance': 1,
            'build_progress': 100,
            'x': 1,
            'y': 1,
            'tag': 0,
        }),
        NamedDict({
            'unit_type': 341,
            'alliance': 1,
            'build_progress': 100,
            'x': 2,
            'y': 2,
            'tag': 0,
        }),
    ]
})

MOVE_LEFT_ACTION = SC2Action.run(actions.RAW_FUNCTIONS.Move_pt,
                        'now', 0, [-1, 1])

class TestCollectablesActionSpace(unittest.TestCase):

    def test_is_action_done(self):
        # GIVEN
        actionSpace = CollectablesActionSpace()
        # WHEN
        is_action_done_return = actionSpace.is_action_done()
        # THEN
        assert is_action_done_return == True

    def test_reset(self):
        # GIVEN
        actionSpace = CollectablesActionSpace()
        # WHEN
        actionSpace.reset()
        #THEN 
        assert actionSpace.move_number == 0
        assert actionSpace.pending_actions == []

    def test_get_actions(self):
        # GIVEN
        actionSpace = CollectablesActionSpace()
        # WHEN
        get_actions_return = actionSpace.get_actions()
        # THEN
        assert get_actions_return == range(0, 4)
        
    def test_get_excluded_actions(self):
        # GIVEN
        actionSpace = CollectablesActionSpace()
        obs = ""
        # WHEN
        get_excluded_actions_return = actionSpace.get_excluded_actions(obs)
        # THEN
        assert get_excluded_actions_return == []

    def test_get_action(self):
        # GIVEN
        actionSpace = CollectablesActionSpace()
        action_idx = 0
        obs = EXAMPLE_OBSERVATION
        # WHEN
        get_action_return = actionSpace.get_action(action_idx, obs)
        # THEN
        assert get_action_return == [actions.RAW_FUNCTIONS.no_op()]

        # GIVEN
        actionSpace = CollectablesActionSpace()
        action_idx = 0
        obs = EXAMPLE_OBSERVATION
        # WHEN
        actionSpace.move_left(obs)
        get_action_return = actionSpace.get_action(action_idx, obs)
        # THEN
        assert get_action_return == [MOVE_LEFT_ACTION]

        # GIVEN
        actionSpace = CollectablesActionSpace()
        action_idx = None
        obs = EXAMPLE_OBSERVATION
        # WHEN
        actionSpace.move_left(obs)
        get_action_return = actionSpace.get_action(action_idx, obs)
        # THEN
        assert get_action_return == [MOVE_LEFT_ACTION]
        assert actionSpace.move_number == 0
        assert actionSpace.pending_actions == []

    def test_solve_action(self):
        # GIVEN
        actionSpace = CollectablesActionSpace()
        action_idx = 0
        obs = EXAMPLE_OBSERVATION
        # WHEN
        actionSpace.solve_action(action_idx, obs)
        # THEN
        assert actionSpace.pending_actions == [MOVE_LEFT_ACTION]

        # GIVEN
        actionSpace = CollectablesActionSpace()
        action_idx = None
        obs = EXAMPLE_OBSERVATION
        # WHEN
        actionSpace.solve_action(action_idx, obs)
        # THEN
        assert actionSpace.move_number == 0
        assert actionSpace.pending_actions == []

    def test_move_left(self):
        # GIVEN
        actionSpace = CollectablesActionSpace()
        obs = EXAMPLE_OBSERVATION
        # WHEN
        actionSpace.move_left(obs)
        # THEN
        assert actionSpace.pending_actions == [MOVE_LEFT_ACTION]

    def test_move_right(self):
        # GIVEN
        actionSpace = CollectablesActionSpace()
        obs = EXAMPLE_OBSERVATION
        # WHEN
        actionSpace.move_right(obs)
        # THEN
        assert actionSpace.pending_actions == [
            SC2Action.run(actions.RAW_FUNCTIONS.Move_pt,
                        'now', 0, [3, 1])]
    
    def test_move_up(self):
        # GIVEN
        actionSpace = CollectablesActionSpace()
        obs = EXAMPLE_OBSERVATION
        # WHEN
        actionSpace.move_up(obs)
        # THEN
        assert actionSpace.pending_actions == [
            SC2Action.run(actions.RAW_FUNCTIONS.Move_pt,
                        'now', 0, [1, -1])]

    def test_move_down(self):
        # GIVEN
        actionSpace = CollectablesActionSpace()
        obs = EXAMPLE_OBSERVATION
        # WHEN
        actionSpace.move_down(obs)
        # THEN
        assert actionSpace.pending_actions == [
            SC2Action.run(actions.RAW_FUNCTIONS.Move_pt,
                        'now', 0, [1, 3])]
        
    def test_get_action_name_str_by_int(self):
        # GIVEN
        actionSpace = CollectablesActionSpace()
        action_int = 0
        # WHEN
        get_action_name_str_by_int_return = \
              actionSpace.get_action_name_str_by_int(action_int)
        assert get_action_name_str_by_int_return == "moveleft"

    def test_get_no_action(self):
        # GIVEN
        actionSpace = CollectablesActionSpace()
        # WHEN
        get_no_action_return = actionSpace.get_no_action()
        # THEN
        assert get_no_action_return == actionSpace.noaction

    def test_get_named_actions(self):
        # GIVEN
        actionSpace = CollectablesActionSpace()
        # WHEN
        get_named_actions_return = actionSpace.get_named_actions()
        # THEN
        assert get_named_actions_return == ['move_left', 'move_right',
                                             'move_up', 'move_down']