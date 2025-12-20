import numpy as np
import unittest
from numpy.testing import assert_array_equal

from experiments.atarinet.simple_env import SimpleRFEnv


class TestSimpleRFEnv(unittest.TestCase):

    def test_step_output_format(self):
        # GIVEN
        env = SimpleRFEnv()

        # WHEN
        env.reset()
        obs, reward, done, truncated, info = env.step(0)

        # THEN
        self.assertEqual(set(obs.keys()), {"screen", "minimap", "nonspatial"})
        self.assertTrue(isinstance(reward, float))
        self.assertTrue(isinstance(done, bool))
        self.assertTrue(isinstance(truncated, bool))
        self.assertTrue(isinstance(info, dict))

    def test_step_moves_player_closer(self):
        # GIVEN
        env = SimpleRFEnv()

        # WHEN
        old_pos = env.player_pos.copy()
        env.step(20)

        # THEN
        new_pos = env.player_pos
        self.assertTrue(not np.allclose(old_pos, new_pos))

    def test_step_reward_negative_by_default(self):
        # GIVEN
        env = SimpleRFEnv()

        # WHEN
        env.target_pos = np.array([1, 1])
        _, reward, done, *_ = env.step(0)

        # THEN
        self.assertEqual(reward, -0.01)
        self.assertTrue(done is False)

    def test_step_reward_when_touching(self):
        # GIVEN
        env = SimpleRFEnv()

        # WHEN
        _, reward, done, *_ = env.step(0)

        # THEN
        self.assertEqual(reward, 1.0)
        self.assertTrue(done)

    def test_render_layer_shapes(self):
        # GIVEN
        env = SimpleRFEnv()

        # WHEN
        screen = env._render_layer(12)
        minimap = env._render_layer(2)

        # THEN
        self.assertEqual(screen.shape, (12, 64, 64))
        self.assertEqual(minimap.shape, (2, 64, 64))


    def test_render_has_player_dot(self):
        # GIVEN
        env = SimpleRFEnv()

        # WHEN
        env.reset()
        layer = env._render_layer(2)

        # THEN
        px = int(env.player_pos[0] * 63)
        py = int(env.player_pos[1] * 63)
        player_channel = layer[0]

        self.assertTrue(player_channel[py, px] > 0)

    def test_render_has_target_square(self):
        # GIVEN
        env = SimpleRFEnv()

        # WHEN
        env.reset()
        layer = env._render_layer(2)

        # THEN
        tx = int(env.target_pos[0] * 63)
        ty = int(env.target_pos[1] * 63)
        target_channel = layer[1]

        # A square of 7x7 should produce non-zero around (tx,ty)
        region = target_channel[max(0, ty-3):ty+4, max(0, tx-3):tx+4]
        self.assertTrue(np.all(region > 0))

    def test_build_nonspatial(self):
        # GIVEN
        env = SimpleRFEnv()

        # WHEN
        env.reset()
        ns = env._build_nonspatial()

        # THEN
        self.assertEqual(ns.shape, (11,))
        self.assertEqual(ns.dtype, np.float32)
        self.assertTrue(np.all((ns[0:4] >= 0) & (ns[0:4] <= 1)))