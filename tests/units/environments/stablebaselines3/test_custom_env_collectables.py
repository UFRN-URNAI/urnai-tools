import unittest
from unittest.mock import ANY, MagicMock, patch

import numpy as np

from urnai.environments.stablebaselines3.custom_env_collectables import (
    CustomEnvCollectables,
)


@patch('urnai.environments.stablebaselines3.custom_env_collectables.CustomEnv.reset')
class TestCustomEnvCollectables(unittest.TestCase):

    def setUp(self):
        self.mock_env_internal = MagicMock()
        self.mock_state = MagicMock()
        self.mock_urnai_action_space = MagicMock()
        self.mock_reward = MagicMock()
        self.mock_observation_space = MagicMock()
        self.mock_action_space = MagicMock()
        self.mock_logger = MagicMock()

        self.mock_urnai_action_space.get_named_actions.return_value = \
            ['action_A', 'action_B']
        self.mock_urnai_action_space.get_actions.return_value = [0, 1]
        
        self.env = CustomEnvCollectables(
            env=self.mock_env_internal,
            state=self.mock_state,
            urnai_action_space=self.mock_urnai_action_space,
            reward=self.mock_reward,
            observation_space=self.mock_observation_space,
            action_space=self.mock_action_space,
            logger=self.mock_logger
        )

    def test_initialization(self, mock_super_reset):
        # GIVEN/WHEN: setUp()
        # THEN
        expected_actions = {0: 'action_A', 1: 'action_B'}
        self.assertDictEqual(self.env.actions, expected_actions)
        self.assertIn('action_A', self.env.action_map_count)
        self.assertEqual(self.env.action_map_count['action_A'], 0)
        self.assertEqual(self.env.step_count, 0)
        
    def test_reset(self, mock_super_reset):
        # GIVEN
        self.env.step_count = 150
        self.env.action_map_count['action_A'] = 5
        # WHEN
        self.env.reset()
        # THEN
        self.assertEqual(self.env.step_count, 0)
        self.assertEqual(self.env.action_map_count['action_A'], 0)
        mock_super_reset.assert_called_once()

    def test_step_basic_flow(self, mock_super_reset):
        # GIVEN
        action_idx = 1 # action_B
        self.mock_env_internal.step.return_value = ("obs_from_env", 1.0, False, False)
        self.mock_state.update.return_value = "updated_obs"
        self.mock_reward.get.return_value = 2.5
        # WHEN
        obs, reward, terminated, truncated, info = self.env.step(action_idx)
        # THEN
        self.mock_urnai_action_space.get_action.assert_called_once_with(action_idx, ANY)
        self.mock_env_internal.step.assert_called_once_with(self.mock_urnai_action_space.get_action.return_value)
        self.mock_state.update.assert_called_once_with("obs_from_env")
        self.mock_reward.get.assert_called_once_with("obs_from_env", 1.0, False, False)
        self.assertEqual(self.env.step_count, self.env.step_mul)
        self.assertEqual(self.env.action_map_count['action_B'], 1)
        self.assertEqual(self.env.action_map_reward['action_B'], 2.5)
        self.assertEqual(obs, "updated_obs")
        self.assertEqual(reward, 2.5)
        self.assertFalse(terminated)

    def test_step_truncates_on_max_steps(self, mock_super_reset):
        # GIVEN
        self.env.max_steps = 100
        self.env.step_count = 100 - self.env.step_mul
        self.env.log_reward_per_action = MagicMock()
        self.mock_env_internal.step.return_value = ("obs", 0, False, False)
        # WHEN
        obs, reward, terminated, truncated, info = self.env.step(0)
        # THEN
        self.assertTrue(truncated)
        self.env.log_reward_per_action.assert_called_once()

    def test_log_reward_per_action_with_logger(self, mock_super_reset):
        # GIVEN
        self.env.action_map_count = {'action_A': 10, 'action_B': 5}
        self.env.action_map_reward = {'action_A': 20.0, 'action_B': 5.0}
        self.env._reward.total_reward = 25.0
        self.env._reward.score = 15
        # WHEN
        self.env.log_reward_per_action()
        # THEN
        self.mock_logger.log.assert_called_once()
        logged_data = self.mock_logger.log.call_args[0][0]
        self.assertAlmostEqual(logged_data["action/avg_reward/action_A"], 2.0)
        self.assertEqual(logged_data["total_reward"], 25.0)
        self.assertEqual(logged_data["shards_collected"], 15)

    def test_log_reward_per_action_without_logger(self, mock_super_reset):
        # GIVEN
        self.env.logger = None
        # WHEN
        self.env.log_reward_per_action()
        # THEN
        self.mock_logger.log.assert_not_called()

    def test_get_action_mask(self, mock_super_reset):
        # GIVEN
        excluded_indices = [1] # 'action_B'
        self.mock_urnai_action_space.get_excluded_actions.return_value =excluded_indices
        self.env.actions = {0: 'action_A', 1: 'action_B', 2: 'action_C'}
        # WHEN
        mask = self.env.get_action_mask()
        # THEN
        expected_mask = np.array([True, False, True], dtype=bool)
        np.testing.assert_array_equal(mask, expected_mask)
        self.mock_urnai_action_space.get_excluded_actions.assert_called_once_with(self.env._obs)