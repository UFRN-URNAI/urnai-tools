import unittest
from unittest.mock import ANY, MagicMock, patch

from urnai.environments.stablebaselines3.custom_env_buildmarines import (
    CustomEnvBuildMarines,
)


@patch('urnai.environments.stablebaselines3.custom_env_buildmarines.CustomEnv.reset')
class TestCustomEnvBuildMarines(unittest.TestCase):

    def setUp(self):
        self.mock_env_internal = MagicMock()
        self.mock_state = MagicMock()
        self.mock_urnai_action_space = MagicMock()
        self.mock_reward = MagicMock()
        self.mock_observation_space = MagicMock()
        self.mock_action_space = MagicMock()
        self.mock_logger = MagicMock()
        self.env_wrapper = CustomEnvBuildMarines(
            env=self.mock_env_internal,
            state=self.mock_state,
            urnai_action_space=self.mock_urnai_action_space,
            reward=self.mock_reward,
            observation_space=self.mock_observation_space,
            action_space=self.mock_action_space,
            logger=self.mock_logger,
            step_mul=32,
            max_steps=5000
        )


    def test_initialization(self, mock_super_reset):
        # GIVEN / WHEN: setUp()
        # THEN
        self.assertIs(self.env_wrapper._env, self.mock_env_internal)
        self.assertIs(self.env_wrapper._state, self.mock_state)
        self.assertIs(self.env_wrapper._action_space, self.mock_urnai_action_space)
        self.assertIs(self.env_wrapper._reward, self.mock_reward)
        self.assertIs(self.env_wrapper.observation_space, self.mock_observation_space)
        self.assertIs(self.env_wrapper.action_space, self.mock_action_space)
        self.assertEqual(self.env_wrapper.step_count, 0)
        self.assertEqual(self.env_wrapper.max_steps, 5000)
        self.assertEqual(self.env_wrapper.action_map_count["BuildMarine"], 0)
        self.assertIn(3, self.env_wrapper.actions)
        self.assertEqual(self.env_wrapper.actions[3], "BuildMarine")


    def test_reset(self, mock_super_reset):
        # GIVEN
        self.env_wrapper.step_count = 100
        self.env_wrapper.action_map_count["Collect"] = 5
        self.env_wrapper.action_map_reward["Collect"] = 10.0
        # WHEN
        self.env_wrapper.reset()
        # THEN
        self.assertEqual(self.env_wrapper.step_count, 0)
        self.assertEqual(self.env_wrapper.action_map_count["Collect"], 0)
        self.assertEqual(self.env_wrapper.action_map_reward["Collect"], 0)
        mock_super_reset.assert_called_once()


    def test_step_basic_flow(self, mock_super_reset):
        # GIVEN
        action = 3  # BuildMarine
        mock_obs = "observation_from_sc2"
        updated_obs = "observation_from_state_builder"
        calculated_reward = 1.5
        self.mock_urnai_action_space.get_action.return_value = "urnai_action"
        self.mock_env_internal.step.return_value = (mock_obs, 1.0, False, False)
        self.mock_state.update.return_value = updated_obs
        self.mock_reward.get.return_value = calculated_reward
        # WHEN
        obs, reward, terminated, truncated, info = self.env_wrapper.step(action)
        # THEN
        self.mock_urnai_action_space.get_action.assert_called_once_with(action, ANY)
        self.mock_env_internal.step.assert_called_once_with("urnai_action")
        self.mock_state.update.assert_called_once_with(mock_obs)
        self.mock_reward.get.assert_called_once_with(mock_obs, 1.0,False,False, action)
        self.assertEqual(self.env_wrapper.step_count, self.env_wrapper.step_mul)
        self.assertEqual(self.env_wrapper.action_map_count["BuildMarine"], 1)
        self.assertEqual(self.env_wrapper.action_map_reward["BuildMarine"], 
                         calculated_reward)
        self.assertEqual(obs, updated_obs)
        self.assertEqual(reward, calculated_reward)
        self.assertFalse(terminated)
        self.assertFalse(truncated)


    def test_step_reaches_max_steps_and_logs(self, mock_super_reset):
        # GIVEN
        self.env_wrapper.max_steps = 100
        self.env_wrapper.step_mul = 32
        self.env_wrapper.step_count = 100 - self.env_wrapper.step_mul
        self.mock_env_internal.step.return_value = ("some_obs", 0.0, False, False)
        self.env_wrapper.log_reward_per_action = MagicMock()
        # WHEN
        obs, reward, terminated, truncated, info = self.env_wrapper.step(0)
        # THEN
        self.assertTrue(truncated)
        self.env_wrapper.log_reward_per_action.assert_called_once()
    
    @patch('urnai.environments.stablebaselines3.custom_env_buildmarines.sc2aux.get_my_units_amount')
    def test_log_reward_per_action(self, mock_get_units, mock_super_reset, 
                            ):
        # GIVEN
        self.env_wrapper.action_map_count = {"Collect": 10, 
                                             "BuildSupplyDepot": 2, 
                                             "BuildBarrack": 1, 
                                             "BuildMarine": 0}
        self.env_wrapper.action_map_reward = {"Collect": 5.0, 
                                              "BuildSupplyDepot": 4.0, 
                                              "BuildBarrack": 1.0, 
                                              "BuildMarine": 0.0}
        self.env_wrapper._reward.total_reward = 10.0
        mock_get_units.side_effect = [
            0, # Marines
            2,  # SupplyDepots
            1   # Barracks
        ]
        # WHEN
        self.env_wrapper.log_reward_per_action()
        # THEN
        self.mock_logger.log.assert_called_once()
        logged_data = self.mock_logger.log.call_args[0][0]
        self.assertEqual(logged_data["action/count/Collect"], 10)
        self.assertAlmostEqual(logged_data["action/avg_reward/Collect"], 0.5)
        self.assertEqual(logged_data["action/count/BuildSupplyDepot"], 2)
        self.assertAlmostEqual(logged_data["action/avg_reward/BuildSupplyDepot"], 2.0)
        self.assertEqual(logged_data["total_reward"], 10.0)
        self.assertEqual(logged_data["marines_built"], 0)
        self.assertEqual(logged_data["supply_depots_built"], 2)
        self.assertEqual(logged_data["barracks_built"], 1)
    
    @patch('urnai.environments.stablebaselines3.custom_env_buildmarines.sc2aux.get_my_units_amount')
    def test_log_reward_per_action_when_logger_is_none(self, mock_get_units, 
                                                       mock_super_reset):
        # GIVEN
        self.env_wrapper.action_map_count = {"Collect": 10, 
                                             "BuildSupplyDepot": 2, 
                                             "BuildBarrack": 1, 
                                             "BuildMarine": 5}
        self.env_wrapper.action_map_reward = {"Collect": 5.0, 
                                              "BuildSupplyDepot": 4.0, 
                                              "BuildBarrack": 3.0, 
                                              "BuildMarine": 10.0}
        self.env_wrapper._reward.total_reward = 22.0
        mock_get_units.side_effect = [20, 2, 1]
        self.env_wrapper.logger = None
        # WHEN
        self.env_wrapper.log_reward_per_action()
        # THEN
        self.mock_logger.log.assert_not_called()