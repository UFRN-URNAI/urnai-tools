import unittest
from unittest.mock import patch

from urnai.environments.stablebaselines3.custom_env import CustomEnv


class TestCustomEnv(unittest.TestCase):

    def test_custom_env_init(self):
        # GIVEN any state
        # WHEN
        env = CustomEnv(None, None, None, None, None, None)
        # THEN
        self.assertEqual(env._env, None)
        self.assertEqual(env._state, None)
        self.assertEqual(env._action_space, None)
        self.assertEqual(env._reward, None)
        self.assertEqual(env._obs, None)
        self.assertEqual(env.action_space, None)
        self.assertEqual(env.observation_space, None)
    
    @patch('urnai.rewards.reward_base.RewardBase')
    @patch('urnai.actions.action_space_base.ActionSpaceBase')
    @patch('urnai.states.state_base.StateBase')
    @patch('urnai.environments.environment_base.EnvironmentBase')
    def test_custom_env_step(self,env_mock, state_mock, action_space_mock, reward_mock):
        # GIVEN
        env = CustomEnv(env_mock, state_mock, action_space_mock, reward_mock, None,None)
        env_mock.step.return_value = (None, None, None, None)
        state_mock.update.return_value = None
        reward_mock.get.return_value = None
        # WHEN
        step_return = env.step(None)
        # THEN
        env_mock.step.assert_called_once()
        state_mock.update.assert_called_once()
        reward_mock.get.assert_called_once()
        self.assertEqual(step_return, (None, None, None, None, {}))
    
    @patch('urnai.environments.environment_base.EnvironmentBase')
    @patch('urnai.states.state_base.StateBase')
    def test_custom_env_reset(self, state_mock, env_mock):
        # GIVEN
        env = CustomEnv(env_mock, state_mock, None, None, None, None)
        env_mock.reset.return_value = None
        state_mock.update.return_value = None
        # WHEN
        reset_return = env.reset()
        # THEN
        env_mock.reset.assert_called_once()
        state_mock.update.assert_called_once()
        self.assertEqual(reset_return, (None, {}))
    
    def test_custom_env_render(self):
        # GIVEN
        env = CustomEnv(None, None, None, None, None, None)

        # WHEN / THEN
        with self.assertRaises(NotImplementedError):
            env.render(None)
    
    @patch('urnai.environments.environment_base.EnvironmentBase')
    def test_custom_env_close(self, env_mock):
        # GIVEN
        env = CustomEnv(env_mock, None, None, None, None, None)
        # WHEN
        close_return = env.close()
        # THEN
        env_mock.close.assert_called_once()
        self.assertEqual(close_return, None)