import unittest

from pysc2.env import sc2_env

from urnai.sc2.environments.sc2environment import SC2Env


class TestSC2Environment(unittest.TestCase):

    def test_default_initialization(self):
        env = SC2Env()
        assert env.map_name == 'Simple64'
        assert env.player_race == sc2_env.Race.random
        assert env.enemy_race == sc2_env.Race.random
        assert env.difficulty == sc2_env.Difficulty.very_easy
        assert env.visualize is False
        assert env.reset_done is True
        assert env.spatial_dim == 16
        assert env.step_mul == 16
        assert env.game_steps_per_episode == 0
        assert env.realtime is False
        assert env.self_play is False
        #TODO: Check players array
        assert isinstance(env.env_instance, sc2_env.SC2Env)
        #TODO: Do we have to check for parameters inside env_instance?
        assert env.terminated is False
        assert env.truncated is False
