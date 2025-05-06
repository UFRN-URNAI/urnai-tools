import unittest
from abc import ABCMeta

from urnai.loggers.logger_base import LoggerBase


class FakeLogger(LoggerBase):
    LoggerBase.__abstractmethods__ = set()

class TestLoggerBase(unittest.TestCase):
    
    def test_set_mode(self):
        # GIVEN
        fake_logger = FakeLogger()
        
        # WHEN
        set_mode_return = fake_logger.set_mode("test_mode")
        
        # THEN
        assert isinstance(LoggerBase, ABCMeta)
        assert set_mode_return is None
    
    def test_not_implemented_log_method(self):
        # GIVEN
        fake_logger = FakeLogger()
        
        # WHEN / THEN
        with self.assertRaises(NotImplementedError):
            fake_logger.log(test_arg = "test_message")