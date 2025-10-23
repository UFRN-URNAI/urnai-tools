import unittest
from unittest.mock import MagicMock, patch

from urnai.environments.stablebaselines3.custom_env_collectables import (
    CustomEnvCollectables,
)


class TestCustomEnvCollectables(unittest.TestCase):

    def setUp(self):
        self.mock_env_internal = MagicMock()
        self.mock_state = MagicMock()
        self.mock_urnai_action_space = MagicMock()
        self.mock_reward = MagicMock()
        self.mock_observation_space = MagicMock()
        self.mock_action_space = MagicMock()
        self.mock_logger = MagicMock()
        
        self.env = CustomEnvCollectables(
            env=self.mock_env_internal,
            state=self.mock_state,
            urnai_action_space=self.mock_urnai_action_space,
            reward=self.mock_reward,
            observation_space=self.mock_observation_space,
            action_space=self.mock_action_space,
            logger=self.mock_logger
        )

    @patch('urnai.environments.stablebaselines3.custom_env_collectables.CustomEnvCollectables.log_results')
    @patch('urnai.environments.stablebaselines3.custom_env_collectables.CustomEnv.step')
    def test_step_not_terminated_nor_truncated(self, mock_super_step, mock_log_results):
        # GIVEN
        action_idx = 1

        mock_super_step.return_value = ("updated_obs", 2.5, False, False, {})
        # WHEN
        obs, reward, terminated, truncated, info = self.env.step(action_idx)
        # THEN
        mock_super_step.assert_called_once_with(action_idx)
        self.assertEqual(obs, "updated_obs")
        self.assertEqual(reward, 2.5)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info, {})
        mock_log_results.assert_not_called()

    @patch('urnai.environments.stablebaselines3.custom_env_collectables.CustomEnvCollectables.log_results')
    @patch('urnai.environments.stablebaselines3.custom_env_collectables.CustomEnv.step')
    def test_step_terminated_true(self, mock_super_step, mock_log_results):
        # GIVEN
        action_idx = 1
        mock_super_step.return_value=("updated_obs", 5.8, True, False, {'key': 'value'})
        # WHEN
        obs, reward, terminated, truncated, info = self.env.step(action_idx)
        # THEN
        mock_super_step.assert_called_once_with(action_idx)
        self.assertEqual(obs, "updated_obs")
        self.assertEqual(reward, 5.8)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info, {'key': 'value'})
        mock_log_results.assert_called_once_with(5.8)
    
    @patch('urnai.environments.stablebaselines3.custom_env_collectables.CustomEnvCollectables.log_results')
    @patch('urnai.environments.stablebaselines3.custom_env_collectables.CustomEnv.step')
    def test_step_truncated_true(self, mock_super_step, mock_log_results):
        # GIVEN
        action_idx = 1
        mock_super_step.return_value=("updated_obs", 7, False, True, {'key': 'value'})
        # WHEN
        obs, reward, terminated, truncated, info = self.env.step(action_idx)
        # THEN
        mock_super_step.assert_called_once_with(action_idx)
        self.assertEqual(obs, "updated_obs")
        self.assertEqual(reward, 7)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info, {'key': 'value'})
        mock_log_results.assert_called_once_with(7)

    def test_log_results_with_logger(self):
        # GIVEN
        self.env._reward.score = 15
        final_reward = 25
        # WHEN
        self.env.log_results(final_reward=final_reward)
        # THEN
        self.mock_logger.log.assert_called_once()
        logged_data = self.mock_logger.log.call_args[0][0]
        self.assertEqual(logged_data["total_reward"], 25)
        self.assertEqual(logged_data["shards_collected"], 15)

    def test_log_results_without_logger(self):
        # GIVEN
        self.env.logger = None
        # WHEN
        self.env.log_results(None)
        # THEN
        self.mock_logger.log.assert_not_called()