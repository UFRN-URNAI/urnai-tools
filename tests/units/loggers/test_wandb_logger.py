import unittest
from unittest.mock import MagicMock, patch

from urnai.loggers.wandb_logger import Mode, WandbLogger


class TestWandbLogger(unittest.TestCase):
    @patch("urnai.loggers.wandb_logger.wandb")
    def test_initialization(self, mock_wandb):

        # GIVEN / WHEN
        logger = WandbLogger()

        # THEN
        self.assertEqual(logger.mode, Mode.TRAIN)
        self.assertEqual(logger.mode_name, ["train", "eval"])
        self.assertEqual(logger.step_name, ["train_step", "eval_step"])
        self.assertEqual(logger.num_steps, [0, 0])
        mock_wandb.define_metric.assert_any_call("train_step")
        mock_wandb.define_metric.assert_any_call("eval_step")
        mock_wandb.define_metric.assert_any_call("train/*", step_metric="train_step")
        mock_wandb.define_metric.assert_any_call("eval/*", step_metric="eval_step")

    @patch("urnai.loggers.wandb_logger.wandb")
    def test_set_mode_false(self, mock_wandb):
        
        # GIVEN
        logger = WandbLogger()

        # WHEN
        logger.set_mode(train=False)

        # THEN
        self.assertEqual(logger.mode, Mode.EVAL)
    
    @patch("urnai.loggers.wandb_logger.wandb")
    def test_set_mode_true(self, mock_wandb):
        
        # GIVEN
        logger = WandbLogger()

        # WHEN
        logger.set_mode(train=True)

        # THEN
        self.assertEqual(logger.mode, Mode.TRAIN)

    @patch("urnai.loggers.wandb_logger.wandb")
    def test_log_train_mode(self, mock_wandb):

        # GIVEN
        mock_wandb.log = MagicMock()
        logger = WandbLogger()

        # WHEN
        logger.set_mode(train=True)
        logger.log({"accuracy": 0.95, "loss": 0.1})

        # THEN
        mock_wandb.log.assert_called_once_with({
            "train_step": 0,
            "train/accuracy": 0.95,
            "train/loss": 0.1
        })
        self.assertEqual(logger.num_steps[0], 1)

    @patch("urnai.loggers.wandb_logger.wandb")
    def test_log_eval_mode(self, mock_wandb):

        # GIVEN
        mock_wandb.log = MagicMock()
        logger = WandbLogger()

        # WHEN
        logger.set_mode(train=False)
        logger.log({"accuracy": 0.85, "loss": 0.2})

        # THEN
        mock_wandb.log.assert_called_once_with({
            "eval_step": 0,
            "eval/accuracy": 0.85,
            "eval/loss": 0.2
        })
        self.assertEqual(logger.num_steps[1], 1)
    
    @patch("urnai.loggers.wandb_logger.wandb")
    def test_log_both_modes(self, mock_wandb):

        # GIVEN
        mock_wandb.log = MagicMock()
        logger = WandbLogger()

        # WHEN
        logger.set_mode(train=True)
        logger.log({"accuracy": 0.95, "loss": 0.1})
        logger.set_mode(train=False)
        logger.log({"accuracy": 0.85, "loss": 0.2})
        logger.set_mode(train=True)
        logger.log({"accuracy": 0.7, "loss": 0.3})

        # THEN
        mock_wandb.log.assert_any_call({
            "train_step": 0,
            "train/accuracy": 0.95,
            "train/loss": 0.1
        })
        mock_wandb.log.assert_any_call({
            "eval_step": 0,
            "eval/accuracy": 0.85,
            "eval/loss": 0.2
        })
        mock_wandb.log.assert_any_call({
            "train_step": 1,
            "train/accuracy": 0.7,
            "train/loss": 0.3
        })
        self.assertEqual(logger.num_steps[0], 2)
        self.assertEqual(logger.num_steps[1], 1)