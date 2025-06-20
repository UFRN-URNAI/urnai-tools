import unittest
from unittest.mock import MagicMock, patch

from urnai.sc2.actions.buildmarines import BuildMarinesActionSpace, scaux, units


class TestBuildMarinesActionSpace(unittest.TestCase):

    def setUp(self):
        self.action_space = BuildMarinesActionSpace()
        self.mock_obs = MagicMock()
        self.action_space.pending_actions = []

    def test_initialization(self):
        # GIVEN / WHEN: setUp()
        # THEN
        self.assertEqual(self.action_space.actions[1], 
                         self.action_space.build_supply_depot)
        self.assertEqual(self.action_space.actions[2], self.action_space.build_barrack)
        self.assertIn('build_marine', self.action_space.named_actions)
        self.assertGreater(len(self.action_space.barrack_coords), 0)

    @patch.object(BuildMarinesActionSpace, 'collect_idle')
    def test_solve_action_collect_idle(self, mock_collect_idle):
        # GIVEN
        action_idx = 0
        # WHEN
        self.action_space.solve_action(action_idx, self.mock_obs)
        # THEN
        mock_collect_idle.assert_called_once_with(self.mock_obs)

    @patch('urnai.sc2.actions.buildmarines.random')
    @patch.object(BuildMarinesActionSpace, 'build_pt')
    def test_solve_action_build_supply(self, mock_build_pt, mock_random):
        # GIVEN
        action_idx = 1
        # WHEN
        self.action_space.solve_action(action_idx, self.mock_obs)
        # THEN
        mock_build_pt.assert_called_once_with(
            self.mock_obs, mock_random.choice.return_value, 
            self.action_space.build_supply_depot
        )
    
    @patch('urnai.sc2.actions.buildmarines.random')
    @patch.object(BuildMarinesActionSpace, 'build_pt')
    def test_solve_action_build_barrack(self, mock_build_pt, mock_random):
        # GIVEN
        action_idx = 2
        # WHEN
        self.action_space.solve_action(action_idx, self.mock_obs)
        # THEN
        mock_build_pt.assert_called_once_with(
            self.mock_obs, mock_random.choice.return_value, 
            self.action_space.build_barrack
        )
    
    @patch.object(BuildMarinesActionSpace, 'build_marine_')
    def test_solve_action_build_marine(self, mock_build_marine):
        # GIVEN
        action_idx = 3
        # WHEN
        self.action_space.solve_action(action_idx, self.mock_obs)
        # THEN
        mock_build_marine.assert_called_once_with(self.mock_obs)
    
    @patch.object(BuildMarinesActionSpace, 'reset')
    def test_solve_action_none(self, mock_reset):
        # GIVEN
        action_idx = None
        # WHEN
        self.action_space.solve_action(action_idx, self.mock_obs)
        # THEN
        mock_reset.assert_called_once()
    
    @patch.object(BuildMarinesActionSpace, 'collect_idle')
    @patch.object(BuildMarinesActionSpace, 'build_pt')
    def test_solve_action_noaction(self, mock_build_pt, mock_collect_idle):
        # GIVEN
        self.action_space.noaction = -1
        action_idx = -1
        # WHEN
        self.action_space.solve_action(action_idx, self.mock_obs)
        # THEN
        mock_collect_idle.assert_not_called()
        mock_build_pt.assert_not_called()
    
    def test_solve_action_raises_error_for_invalid_index(self):
        # GIVEN
        invalid_action_idx = 99
        # WHEN / THEN
        with self.assertRaises(ValueError):
            self.action_space.solve_action(invalid_action_idx, self.mock_obs)

    @patch('urnai.sc2.actions.buildmarines.sc2_actions')
    @patch('urnai.sc2.actions.buildmarines.random')
    @patch('urnai.sc2.actions.buildmarines.scaux')
    def test_collect_idle_with_idle_worker(self, mock_scaux, mock_random, 
                                           mock_sc2_actions):
        # GIVEN
        mock_scv = MagicMock()
        mock_scv.tag = 101
        mock_mineral = MagicMock()
        mock_mineral.tag = 202
        mock_scaux.get_random_idle_worker.return_value = mock_scv
        mock_scaux.get_neutral_units_by_type.return_value = [mock_mineral]
        mock_random.choice.return_value = mock_mineral
        mock_harvest_action = MagicMock()
        mock_sc2_actions.__getitem__.return_value = mock_harvest_action
        # WHEN
        self.action_space.collect_idle(self.mock_obs)
        # THEN
        self.assertEqual(len(self.action_space.pending_actions), 1)
        mock_harvest_action.run.assert_called_once_with('queued', 101, 202)
    
    @patch('urnai.sc2.actions.buildmarines.sc2_actions')
    @patch('urnai.sc2.actions.buildmarines.random')
    @patch('urnai.sc2.actions.buildmarines.scaux')
    def test_collect_idle_with_no_minerals(self, mock_scaux, mock_random, 
                                           mock_sc2_actions):
        # GIVEN
        mock_scv = MagicMock()
        mock_scv.tag = 101
        mock_scaux.get_random_idle_worker.return_value = mock_scv
        mock_scaux.get_neutral_units_by_type.return_value = []
        # WHEN
        self.action_space.collect_idle(self.mock_obs)
        # THEN
        self.assertEqual(len(self.action_space.pending_actions), 0)

    @patch('urnai.sc2.actions.buildmarines.scaux')
    def test_collect_idle_with_no_idle_worker(self, mock_scaux):
        # GIVEN
        mock_scaux.get_random_idle_worker.return_value = scaux._NO_UNITS
        mock_scaux._NO_UNITS = scaux._NO_UNITS
        # WHEN
        self.action_space.collect_idle(self.mock_obs)
        # THEN
        self.assertEqual(len(self.action_space.pending_actions), 0)

    @patch('urnai.sc2.actions.buildmarines.sc2_actions')
    @patch.object(BuildMarinesActionSpace, 'select_random_scv')
    def test_build_pt(self, mock_select_scv, mock_sc2_actions):
        # GIVEN
        mock_scv = MagicMock()
        mock_scv.tag = 101
        mock_select_scv.return_value = mock_scv
        coords = {'x': 42, 'y': 43}
        action_name = self.action_space.build_supply_depot
        mock_build_action = MagicMock()
        mock_sc2_actions.__getitem__.return_value = mock_build_action
        # WHEN
        self.action_space.build_pt(self.mock_obs, coords, action_name)
        # THEN
        self.assertEqual(len(self.action_space.pending_actions), 1)
        mock_build_action.run.assert_called_once_with('now', 101, [42, 43])
    
    @patch.object(BuildMarinesActionSpace, 'select_random_scv')
    def test_build_pt_no_scv_available(self, mock_select_scv):
        # GIVEN
        mock_select_scv.return_value = None
        coords = {'x': 42, 'y': 43}
        action_name = self.action_space.build_supply_depot
        # WHEN
        self.action_space.build_pt(self.mock_obs, coords, action_name)
        # THEN
        self.assertEqual(len(self.action_space.pending_actions), 0)

    @patch('urnai.sc2.actions.buildmarines.sc2_actions')
    @patch('urnai.sc2.actions.buildmarines.random')
    @patch('urnai.sc2.actions.buildmarines.scaux')
    def test_build_marine_when_barracks_exist(self, mock_scaux, mock_random, 
                                              mock_sc2_actions):
        # GIVEN
        mock_barrack = MagicMock()
        mock_barrack.tag = 303
        mock_scaux.get_units_by_type.return_value = [mock_barrack]
        mock_random.choice.return_value = mock_barrack
        mock_train_action = MagicMock()
        mock_sc2_actions.__getitem__.return_value = mock_train_action
        # WHEN
        self.action_space.build_marine_(self.mock_obs)
        # THEN
        self.assertEqual(len(self.action_space.pending_actions), 1)
        mock_train_action.run.assert_called_once_with('now', 303)

    @patch('urnai.sc2.actions.buildmarines.scaux')
    def test_build_marine_when_no_barracks(self, mock_scaux):
        # GIVEN
        mock_scaux.get_units_by_type.return_value = []
        # WHEN
        self.action_space.build_marine_(self.mock_obs)
        # THEN
        self.assertEqual(len(self.action_space.pending_actions), 0)
    
    @patch('urnai.sc2.actions.buildmarines.random')
    @patch('urnai.sc2.actions.buildmarines.scaux')
    def test_select_random_scv_returns_correct_unit(self, mock_scaux, mock_random):
        # GIVEN
        mock_scv1 = MagicMock()
        mock_scv2 = MagicMock()
        mock_scv3 = MagicMock()
        fake_scv_list = [mock_scv1, mock_scv2, mock_scv3]
        mock_scaux.get_units_by_type.return_value = fake_scv_list
        mock_random.randint.return_value = 1
        # WHEN
        selected_scv = self.action_space.select_random_scv(self.mock_obs)
        # THEN
        self.assertIs(selected_scv, mock_scv2)
        mock_scaux.get_units_by_type.assert_called_once_with(self.mock_obs, 
                                                             units.Terran.SCV)
        mock_random.randint.assert_called_once_with(0, 2)


    @patch('urnai.sc2.actions.buildmarines.scaux')
    def test_select_random_scv_with_no_scvs(self, mock_scaux):
        # GIVEN
        mock_scaux.get_units_by_type.return_value = []
        # WHEN
        result = self.action_space.select_random_scv(self.mock_obs)
        # THEN
        self.assertIsNone(result)