import unittest
from unittest.mock import Mock

from experiments.atarinet.quicklog import QuickLog


class TestQuickLog(unittest.TestCase):

    def test_log_during_ep_new_action(self):
        # GIVEN
        log = QuickLog()

        # WHEN
        log.log_during_ep("move", 5)

        # THEN
        self.assertEqual(log.model_stats["total_reward"], 5)
        self.assertEqual(log.model_stats["actions"], {"move": 1})

    def test_log_during_ep_existing_action(self):
        # GIVEN
        log = QuickLog()

        # WHEN
        log.log_during_ep("move", 2)
        log.log_during_ep("move", 3)

        # THEN
        self.assertEqual(log.model_stats["total_reward"], 5)
        self.assertEqual(log.model_stats["actions"]["move"], 2)

    def test_end_of_ep_resets(self):
        # GIVEN
        log = QuickLog()

        # WHEN
        log.log_during_ep("move", 10)
        log.end_of_ep()

        # THEN
        self.assertEqual(log.model_stats["total_reward"], 0)
        self.assertEqual(log.model_stats["actions"], {})

    def test_wandb_log(self):
        # GIVEN
        log = QuickLog()

        log.model_stats = {
            "total_reward": 42,
            "actions": {"move": 3, "attack": 7}
        }

        model = Mock()
        model.total_loss = 100
        model.replay_buffer = [1, 2, 3]  # len = 3
        model.epsilon = 0.12

        wandb_logger = Mock()

        # WHEN
        log.wandb_log(model, step=5, ep=2, wandb_logger=wandb_logger)

        # THEN
        wandb_logger.log.assert_called_once()
        logged_data = wandb_logger.log.call_args[0][0]

        self.assertEqual(logged_data["mean loss"], 100 / 5)
        self.assertEqual(logged_data["reward per ep"], 42)
        self.assertEqual(logged_data["ep length"], 5)
        self.assertEqual(logged_data["episode"], 2)
        self.assertEqual(logged_data["replay_buffer_len"], 3)
        self.assertEqual(logged_data["epsilon"], 0.12)

    def test_wandb_log_step_zero_no_division_by_zero(self):
        # GIVEN
        log = QuickLog()

        log.model_stats = {
            "total_reward": 0,
            "actions": {}
        }

        model = Mock()
        model.total_loss = 50
        model.replay_buffer = []
        model.epsilon = 0.5

        wandb_logger = Mock()

        # WHEN
        log.wandb_log(model, step=0, ep=1, wandb_logger=wandb_logger)

        # THEN
        logged = wandb_logger.log.call_args[0][0]
        self.assertEqual(logged["mean loss"], 50 / 1)
