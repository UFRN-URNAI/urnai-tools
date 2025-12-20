import io
import os
import tempfile
import unittest
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import torch


from urnai.sc2.models.atarinet_model import ReplayMemory, AtariNetModel, Transition, DeepmindState


def make_dummy_state():
    """Create a single state tuple (screen, minimap, nonspatial) like your ReplayMemory expects."""
    screen = np.zeros((4, 12, 64, 64), dtype=np.float32)  # n_frame_stack=4, channels=12
    minimap = np.zeros((4, 2, 64, 64), dtype=np.float32)
    nonspatial = np.zeros((4, 11), dtype=np.float32)
    return (screen, minimap, nonspatial)


class TestReplayMemory(unittest.TestCase):
    def test_push_and_len_and_sample(self):
        # GIVEN
        mem = ReplayMemory(5)

        # WHEN
        for i in range(3):
            s = f"s{i}"
            a = i
            r = float(i)
            ns = f"ns{i}"
            mem.push(s, a, r, ns)

        # THEN
        self.assertEqual(len(mem), 3)
        sample = mem.sample(2)
        self.assertEqual(len(sample), 2)
        self.assertTrue(all(isinstance(t, Transition) for t in sample))


class TestAtariNetModelWithPatches(unittest.TestCase):
    def setUp(self):

        self.map_name = "DummyMap"
        self.replay_buffer_size = 10
        self.batch_size = 4

        self.model = AtariNetModel(
            map_name=self.map_name,
            replay_buffer_size=self.replay_buffer_size,
            batch_size=self.batch_size,
        )

    def test_model_initialization_creates_networks_and_buffer(self):
        # GIVEN

        # THEN
        self.assertIsNotNone(self.model.policy_net)
        self.assertIsNotNone(self.model.target_net)
        self.assertIsNotNone(self.model.replay_buffer)
        self.assertEqual(len(self.model.replay_buffer), 0)

        self.assertEqual(len(self.model.available_actions), 22 * 16)

    def test_to_DeepmindState_returns_correct_tensors(self):
        # GIVEN
        s1 = make_dummy_state()
        s2 = make_dummy_state()
        batch = [s1, s2]

        # WHEN
        deep = self.model.to_DeepmindState(batch)

        # THEN
        self.assertTrue(isinstance(deep.screen, torch.Tensor))
        self.assertTrue(isinstance(deep.minimap, torch.Tensor))
        self.assertTrue(isinstance(deep.nonspatial, torch.Tensor))

        # shapes: batch, channels, 64,64 and batch, channels_nonspatial
        # n_frame_stack default is 4 -> channels_screen = 4*12 = 48
        self.assertEqual(deep.screen.shape[0], 2)
        self.assertEqual(deep.screen.shape[2], 64)
        self.assertEqual(deep.minimap.shape[0], 2)
        self.assertEqual(deep.nonspatial.shape[0], 2)

    def test_make_frame_stack_and_clear(self):
        # GIVEN
        frame = (
            np.ones((12, 64, 64), dtype=np.float32),  # screen single frame (channels, H, W)
            np.ones((2, 64, 64), dtype=np.float32),   # minimap
            np.ones((11,), dtype=np.float32),         # nonspatial
        )

        # WHEN
        stacked = self.model.make_frame_stack(frame)

        # THEN
        self.assertEqual(len(stacked), 3)
        self.assertEqual(stacked[0].shape[0], self.model.n_frame_stack)  # screen stacked frames
        self.assertEqual(stacked[1].shape[0], self.model.n_frame_stack)  # minimap stacked
        self.assertEqual(stacked[2].shape[0], self.model.n_frame_stack)  # nonspatial stacked

        self.model.clear_frame_stack()
        for dq in self.model.frame_stack:
            self.assertEqual(len(dq), 0)

    def test_new_ep_resets_total_loss_and_decays_epsilon(self):
        # GIVEN
        old_epsilon = self.model.epsilon
        self.model.total_loss = 123.4

        # WHEN
        self.model.new_ep()

        # THEN
        self.assertEqual(self.model.total_loss, 0)
        self.assertLessEqual(self.model.epsilon, old_epsilon)
        self.assertGreaterEqual(self.model.epsilon, self.model.epsilon_min)

    def test_epsilon_decay_reaches_minimum(self):
        # GIVEN
        self.model.epsilon = 0.02
        for _ in range(1000):
            self.model.epsilon_decay()
        self.assertGreaterEqual(self.model.epsilon, self.model.epsilon_min)
        self.assertLessEqual(self.model.epsilon, 0.02)

    def test_predict_exploration_and_greedy(self):
        # GIVEN
        self.model.training = True
        self.model.epsilon = 1.0
        # WHEN
        action, args = self.model.predict(make_dummy_state())
        # THEN
        self.assertIn(action, self.model.available_actions)
        self.assertEqual(args, [])

        # GIVEN
        self.model.training = False

        fake_out = torch.zeros(1, len(self.model.available_actions))
        fake_out[0, 5] = 1.0
        self.model.policy_net = MagicMock(return_value=fake_out)

        # WHEN
        act2, args2 = self.model.predict(make_dummy_state())

        # THEN
        self.assertEqual(act2, 5)
        self.assertEqual(args2, [])

    def test_learn_pushes_to_replay_and_calls_optimize_when_enough(self):
        # GIVEN
        self.model.batch_size = 2

        self.model.optimize_model = Mock()
        self.model.soft_update = Mock()

        s = make_dummy_state()
        # WHEN
        self.model.learn(s, 1, 0.0, s, False)
        self.model.learn(s, 2, 0.0, s, False)

        # THEN
        self.assertTrue(self.model.optimize_model.called)
        self.assertTrue(self.model.soft_update.called)

    @patch("urnai.sc2.models.atarinet_model.torch.load")
    def test_load_restores_epsilon_and_recreates_optimizer(self, mock_torch_load):
        # GIVEN
        fake_state = {
            "policy_net": self.model.policy_net.state_dict(),
            "target_net": self.model.target_net.state_dict(),
            "epsilon": 0.321
        }
        mock_torch_load.return_value = fake_state

        # WHEN
        tmpfile = tempfile.NamedTemporaryFile(delete=False)
        path = tmpfile.name
        tmpfile.close()

        self.model.load(path)

        # THEN
        self.assertEqual(self.model.epsilon, 0.321)
        self.assertIsNotNone(self.model.optimizer)

        os.remove(path)
