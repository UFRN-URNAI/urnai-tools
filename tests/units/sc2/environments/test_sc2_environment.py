import unittest
from unittest.mock import patch

from urnai.sc2.environments.sc2environment import SC2Env


class TestSC2Environment(unittest.TestCase):
    @patch('pysc2.lib.features.Dimensions')
    @patch('pysc2.lib.features.AgentInterfaceFormat')
    @patch('pysc2.env.sc2_env.SC2Env')
    def test_start_on_init(self, MockSC2Env, MockAgentInterfaceFormat, MockDimensions):
        SC2Env()
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
        env = SC2Env()
        env.start()
        MockSC2Env.assert_called_once()
        MockAgentInterfaceFormat.assert_called_once()
        MockDimensions.assert_called_once()

    def test_step_method(self):
        # TODO
        pass

    def test_reset_method(self):
        # TODO
        pass

    @patch('pysc2.env.sc2_env.SC2Env')
    def test_close_method(self, MockSC2Env):
        # TODO: Check why this does not work
        # env = SC2Env()
        # env.close()
        # MockSC2Env.close.assert_called_once()
        pass
    
    def test_restart_method(self):
        # TODO
        pass

    def test_parse_timestep_method(self):
        # TODO
        pass
