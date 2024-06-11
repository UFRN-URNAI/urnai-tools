import unittest
from unittest.mock import patch

from pysc2.env import sc2_env
from pysc2.env.environment import StepType, TimeStep
from pysc2.lib import actions, point
from pysc2.lib.named_array import NamedDict

from urnai.sc2.environments.sc2environment import SC2Env


class TestSC2Environment(unittest.TestCase):
    @patch('pysc2.lib.features.Dimensions')
    @patch('pysc2.lib.features.AgentInterfaceFormat')
    @patch('pysc2.env.sc2_env.SC2Env')
    def test_start_on_init(self, MockSC2Env, MockAgentInterfaceFormat, MockDimensions):
        # GIVEN any state
        # WHEN
        SC2Env(players=[
                sc2_env.Agent(sc2_env.Race.protoss),
                sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy),
            ])
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
        env = SC2Env(players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.easy),
            ])
        # WHEN
        env.start()
        # THEN
        MockSC2Env.assert_called_once()
        MockAgentInterfaceFormat.assert_called_once()
        MockDimensions.assert_called_once()

    @patch('pysc2.env.sc2_env.SC2Env')
    def test_reset_method(self, scenv_mock):
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
        env.env_instance.reset.return_value = fakeTimestep
        # WHEN
        obs = env.reset()
        # THEN
        env.env_instance.reset.assert_called_once()
        assert isinstance(obs, NamedDict)
    
    @patch('pysc2.env.sc2_env.SC2Env')
    def test_step_method(self, scenv_mock):
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
        env.env_instance.step.return_value = fakeTimestep
        # WHEN
        obs, reward, terminated, truncated = env.step([actions.RAW_FUNCTIONS.no_op()])
        # THEN
        env.env_instance.step.assert_called_once_with([actions.RAW_FUNCTIONS.no_op()])
        assert isinstance(obs, NamedDict)
        assert obs.step_mul == 16
        assert isinstance(obs.map_size, point.Point)
        assert obs.map_size.x == 64
        assert obs.map_size.y == 64
        assert reward == 0
        assert terminated is False
        assert truncated is False


    @patch('pysc2.env.sc2_env.SC2Env')
    def test_close_method(self, scenv_mock):
        # GIVEN
        env = SC2Env()
        # WHEN
        env.close()
        # THEN
        env.env_instance.close.assert_called_once()
    
    @patch('pysc2.env.sc2_env.SC2Env')
    def test_restart_method(self, scenv_mock):
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
        env.env_instance.reset.return_value = fakeTimestep
        # WHEN
        env.restart()
        # THEN
        env.env_instance.close.assert_called_once()
        env.env_instance.reset.assert_called_once()

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
        assert isinstance(obs, NamedDict)
        assert obs.step_mul == 16
        assert isinstance(obs.map_size, point.Point)
        assert obs.map_size.x == 64
        assert obs.map_size.y == 64
        assert reward == 0
        assert terminated is False
        assert truncated is False
