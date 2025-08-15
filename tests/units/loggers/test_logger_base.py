import unittest
from abc import ABCMeta
from unittest.mock import patch, call

from urnai.logging.logging_mode_base import LoggingModeBase

from urnai.logging.logger_base import LoggerBase
from urnai.logging.logging_mode_base import LoggingModeBase
from typing import get_type_hints


class FakeModeCls(LoggingModeBase):
    TRAINING = "training"
    EVALUATION = "evaluation"


class FakeLogger(LoggerBase):
    LoggerBase.__abstractmethods__ = set()
    _mode_cls: FakeModeCls = FakeModeCls


class TestLoggerBase(unittest.TestCase):

    @patch('urnai.logging.logger_base.LoggingModeBase.check')
    def test_set_mode_calls_check(self, mock_check):
        # GIVEN
        mock_check.return_value = None
        fake_logger = FakeLogger("training")

        # WHEN
        fake_logger.set_mode("evaluation")

        # THEN
        mock_check.assert_has_calls([
            call("training"),
            call("evaluation"),
        ])

    def test_not_implemented_log_method(self):
        # GIVEN
        fake_logger = FakeLogger("training")

        # WHEN / THEN
        with self.assertRaises(NotImplementedError):
            fake_logger.log(test_arg = "test_message")
