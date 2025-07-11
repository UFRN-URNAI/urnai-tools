import unittest
from unittest.mock import MagicMock, patch

from urnai.sc2.actions.collectables import CollectablesActionSpace


def make_mock_unit(x, y, tag):
    unit = MagicMock()
    unit.x = x
    unit.y = y
    unit.tag = tag
    return unit

class TestCollectablesActionSpace(unittest.TestCase):

    def setUp(self):
        self.action_space = CollectablesActionSpace()
        self.mock_obs = MagicMock()

    def test_reset_clears_state(self):
        # GIVEN
        self.action_space.move_number = 5
        self.action_space.pending_actions.append("action")
        # WHEN
        self.action_space.reset()
        # THEN
        self.assertEqual(self.action_space.move_number, 0)
        self.assertEqual(self.action_space.pending_actions, [])

    @patch('urnai.sc2.actions.collectables.scaux.select_army')
    def test_is_move_valid_when_all_moves_are_possible(self, mock_select_army):
        # GIVEN
        mock_army = [make_mock_unit(30, 35, 1)]
        mock_select_army.return_value = mock_army
        # WHEN/THEN
        self.assertTrue(self.action_space.is_move_valid(self.mock_obs, 
                                                        self.action_space.move_left))
        self.assertTrue(self.action_space.is_move_valid(self.mock_obs, 
                                                        self.action_space.move_right))
        self.assertTrue(self.action_space.is_move_valid(self.mock_obs, 
                                                        self.action_space.move_up))
        self.assertTrue(self.action_space.is_move_valid(self.mock_obs, 
                                                        self.action_space.move_down))

    @patch('urnai.sc2.actions.collectables.scaux.select_army')
    def test_is_move_valid_when_at_left_wall(self, mock_select_army):
        # GIVEN
        mock_army = [make_mock_unit(22, 35, 1)]
        mock_select_army.return_value = mock_army
        
        # WHEN/THEN
        self.assertFalse(self.action_space.is_move_valid(self.mock_obs, 
                                                         self.action_space.move_left))
        self.assertTrue(self.action_space.is_move_valid(self.mock_obs, 
                                                        self.action_space.move_right))

    @patch.object(CollectablesActionSpace, 'is_move_valid')
    def test_get_excluded_actions(self, mock_is_move_valid):
        # GIVEN
        mock_is_move_valid.side_effect = lambda obs, action: action not in [
            self.action_space.move_left, self.action_space.move_down
        ]
        # WHEN
        excluded = self.action_space.get_excluded_actions(self.mock_obs)
        # THEN
        self.assertCountEqual(excluded, [self.action_space.move_left, 
                                         self.action_space.move_down])

    @patch('urnai.sc2.actions.collectables.sc2_actions')
    @patch('urnai.sc2.actions.collectables.scaux.select_army')
    def test_move_left_calculates_correct_coordinates(self, mock_select_army, 
                                                      mock_sc2_actions):
        # GIVEN
        mock_army = [make_mock_unit(30, 40, 1), make_mock_unit(32, 40, 2)]
        mock_select_army.return_value = mock_army
        mock_move_action = MagicMock()
        mock_sc2_actions.__getitem__.return_value = mock_move_action
        # WHEN
        self.action_space.move_left_(self.mock_obs)
        # THEN
        self.assertEqual(len(self.action_space.pending_actions), 2)
        expected_coords = [29, 40]
        mock_move_action.run.assert_any_call('now', 1, expected_coords)
        mock_move_action.run.assert_any_call('now', 2, expected_coords)

    @patch.object(CollectablesActionSpace, 'move_left_')
    def test_solve_action_dispatches_to_move_left(self, mock_move_left):
        # GIVEN
        action_idx = self.action_space.move_left
        # WHEN
        self.action_space.solve_action(action_idx, self.mock_obs)
        # THEN
        mock_move_left.assert_called_once_with(self.mock_obs)
        
    @patch.object(CollectablesActionSpace, 'solve_action')
    def test_get_action_returns_no_op_when_pending_is_empty(self, mock_solve_action):
        # GIVEN
        self.action_space.pending_actions = []
        # WHEN
        action_returned = self.action_space.get_action(0, self.mock_obs)
        # THEN
        self.assertEqual(action_returned, self.action_space.noaction)
        mock_solve_action.assert_called_once_with(0, self.mock_obs)

    @patch.object(CollectablesActionSpace, 'solve_action')
    def test_get_action_returns_pending_action(self, mock_solve_action):
        # GIVEN
        action1 = "action_1"
        action2 = "action_2"
        self.action_space.pending_actions = [action1, action2]
        # WHEN
        action_returned = self.action_space.get_action(0, self.mock_obs)
        # THEN
        self.assertEqual(action_returned, [action2])
        self.assertEqual(len(self.action_space.pending_actions), 1)
        self.assertEqual(self.action_space.pending_actions[0], action1)

    def test_is_action_done(self):
        # GIVEN / WHEN
        self.action_space.pending_actions = []
        # THEN
        self.assertTrue(self.action_space.is_action_done())
        # GIVEN / WHEN
        self.action_space.pending_actions = ["action"]
        # THEN
        self.assertFalse(self.action_space.is_action_done())

    def test_simple_getters(self):
        # GIVEN/WHEN: setUp()
        # THEN
        self.assertEqual(self.action_space.get_actions(), range(4))
        expected_names = ['move_left', 'move_right', 'move_up', 'move_down']
        self.assertEqual(self.action_space.get_named_actions(), expected_names)

    @patch.object(CollectablesActionSpace, 'move_right_')
    def test_solve_action_dispatches_to_move_right(self, mock_move_right):
        # GIVEN : setUp()
        # WHEN
        self.action_space.solve_action(self.action_space.move_right, self.mock_obs)
        # THEN
        mock_move_right.assert_called_once_with(self.mock_obs)

    @patch.object(CollectablesActionSpace, 'move_up_')
    def test_solve_action_dispatches_to_move_up(self, mock_move_up):
        # GIVEN : setUp()
        # WHEN
        self.action_space.solve_action(self.action_space.move_up, self.mock_obs)
        # THEN
        mock_move_up.assert_called_once_with(self.mock_obs)

    @patch.object(CollectablesActionSpace, 'move_down_')
    def test_solve_action_dispatches_to_move_down(self, mock_move_down):
        # GIVEN : setUp()
        # WHEN
        self.action_space.solve_action(self.action_space.move_down, self.mock_obs)
        # THEN
        mock_move_down.assert_called_once_with(self.mock_obs)

    @patch.object(CollectablesActionSpace, 'reset')
    def test_solve_action_dispatches_to_reset(self, mock_reset):
        # GIVEN : setUp()
        # WHEN
        self.action_space.solve_action(None, self.mock_obs)
        # THEN
        mock_reset.assert_called_once()

    @patch('urnai.sc2.actions.collectables.sc2_actions')
    @patch('urnai.sc2.actions.collectables.scaux.select_army')
    def test_move_right_calculates_correct_coordinates(self, mock_select_army, 
                                                       mock_sc2_actions):
        # GIVEN
        mock_army = [make_mock_unit(30, 40, 1)]
        mock_select_army.return_value = mock_army
        mock_move_action = MagicMock()
        mock_sc2_actions.__getitem__.return_value = mock_move_action
        # WHEN
        self.action_space.move_right_(self.mock_obs)
        # THEN
        expected_coords = [32, 40]
        mock_move_action.run.assert_called_once_with('now', 1, expected_coords)
        
    @patch('urnai.sc2.actions.collectables.sc2_actions')
    @patch('urnai.sc2.actions.collectables.scaux.select_army')
    def test_move_up_calculates_correct_coordinates(self, mock_select_army, 
                                                    mock_sc2_actions):
        # GIVEN
        mock_army = [make_mock_unit(30, 40, 1)]
        mock_select_army.return_value = mock_army
        mock_move_action = MagicMock()
        mock_sc2_actions.__getitem__.return_value = mock_move_action
        # WHEN
        self.action_space.move_up_(self.mock_obs)
        # THEN
        expected_coords = [30, 38]
        mock_move_action.run.assert_called_once_with('now', 1, expected_coords)

    @patch('urnai.sc2.actions.collectables.sc2_actions')
    @patch('urnai.sc2.actions.collectables.scaux.select_army')
    def test_move_down_calculates_correct_coordinates(self, mock_select_army, 
                                                      mock_sc2_actions):
        # GIVEN
        mock_army = [make_mock_unit(30, 40, 1)]
        mock_select_army.return_value = mock_army
        mock_move_action = MagicMock()
        mock_sc2_actions.__getitem__.return_value = mock_move_action
        # WHEN
        self.action_space.move_down_(self.mock_obs)
        # THEN
        expected_coords = [30, 42]
        mock_move_action.run.assert_called_once_with('now', 1, expected_coords)
    
    @patch.object(CollectablesActionSpace, 'move_left_')
    @patch.object(CollectablesActionSpace, 'move_right_')
    @patch.object(CollectablesActionSpace, 'move_up_')
    @patch.object(CollectablesActionSpace, 'move_down_')
    def test_solve_action_does_nothing_for_noaction(self, mock_move_down, mock_move_up, 
                                                    mock_move_right, mock_move_left):
        # GIVEN
        no_action_obj = self.action_space.noaction
        # WHEN
        self.action_space.solve_action(no_action_obj, self.mock_obs)
        # THEN
        mock_move_left.assert_not_called()
        mock_move_right.assert_not_called()
        mock_move_up.assert_not_called()
        mock_move_down.assert_not_called()
    
    def test_solve_action_raises_error_for_invalid_index(self):
        # GIVEN
        invalid_action_idx = 99
        # WHEN / THEN
        with self.assertRaises(ValueError):
            self.action_space.solve_action(invalid_action_idx, self.mock_obs)