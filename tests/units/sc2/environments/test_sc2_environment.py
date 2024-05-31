import unittest
from unittest.mock import patch

from pysc2.env.environment import StepType, TimeStep
from pysc2.lib import point
from pysc2.lib.named_array import NamedDict

from urnai.sc2.environments.sc2environment import SC2Env


class TestSC2Environment(unittest.TestCase):
    @patch('pysc2.lib.features.Dimensions')
    @patch('pysc2.lib.features.AgentInterfaceFormat')
    @patch('pysc2.env.sc2_env.SC2Env')
    def test_start_on_init(self, MockSC2Env, MockAgentInterfaceFormat, MockDimensions):
        # GIVEN any state
        # WHEN
        SC2Env()
        # THEN
        MockSC2Env.assert_called_once()
        MockAgentInterfaceFormat.assert_called_once()
        MockDimensions.assert_called_once()

    @patch('pysc2.lib.features.Dimensions')
    @patch('pysc2.lib.features.AgentInterfaceFormat')
    @patch('pysc2.env.sc2_env.SC2Env')
    def test_start_after_init(
        self, MockSC2Env, MockAgentInterfaceFormat, MockDimensions
    ):
        # Even after another call of start, the sc2_env of pysc2
        # should only be instanciated once.

        # GIVEN
        env = SC2Env()
        # WHEN
        env.start()
        # THEN
        MockSC2Env.assert_called_once()
        MockAgentInterfaceFormat.assert_called_once()
        MockDimensions.assert_called_once()

    @patch('pysc2.env.sc2_env.SC2Env')
    def test_reset_method(self, scenv_mock):
        # GIVEN
        env = SC2Env(step_mul=16, game_steps_per_episode=100)
        env.env_instance._episode_steps = 16
        env.env_instance._episode_length = 100
        # WHEN
        obs = env.reset()
        # THEN
        env.env_instance.reset.assert_called_once()
        isinstance(obs, NamedDict)
        pass
    
    # @patch('pysc2.env.sc2_env.SC2Env')
    # def test_step_method(self, scenv_mock):
    #     # TODO
    #     # GIVEN
    #     env = SC2Env()
    #     env.reset()
    #     # WHEN
    #     obs, reward, terminated, truncated = env.step([actions.RAW_FUNCTIONS.no_op()])
    #     # THEN
    #     env.env_instance.step.assert_called_once()
    #     pass


    @patch('pysc2.env.sc2_env.SC2Env')
    def test_close_method(self, scenv_mock):
        # GIVEN
        env = SC2Env()
        # WHEN
        env.close()
        # THEN
        env.env_instance.close.assert_called_once()
    
    def test_restart_method(self):
        # TODO
        # GIVEN
        # WHEN
        # THEN
        pass

    @patch('pysc2.env.sc2_env.SC2Env')
    def test_parse_timestep_method(self, scenv_mock):

        # GIVEN
        env = SC2Env(map_name='Simple64', step_mul=16, game_steps_per_episode=100)
        fakeTimestep = ( TimeStep( 
            step_type= StepType.FIRST, 
            reward=0, 
            discount=0, 
            observation=NamedDict({})), 
        )
        env.env_instance._episode_steps = 16
        env.env_instance._episode_length = 100
        env.env_instance._interface_formats[0]._raw_resolution = point.Point(64,64)
        # WHEN
        obs, reward, terminated, truncated = env._parse_timestep(fakeTimestep)
        # THEN
        isinstance(obs, NamedDict)
        assert obs.step_mul == 16
        isinstance(obs.map_size, point.Point)
        assert obs.map_size.x == 64
        assert obs.map_size.y == 64
        assert reward == 0
        assert terminated is False
        assert truncated is False
