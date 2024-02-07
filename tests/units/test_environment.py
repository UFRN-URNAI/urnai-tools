from abc import ABCMeta
from urnai.environments.environment_base import EnvironmentBase
import unittest

class TestEnvironmentBase(unittest.TestCase):

    def test_abstract_methods(self):
        EnvironmentBase.__abstractmethods__ = set()

        class FakeEnvironment(EnvironmentBase):
            def __init__(self, id: str, render=False, reset_done=True):
                super().__init__(id, render, reset_done)

        f = FakeEnvironment("id")
        start_return = f.start()
        step_return = f.step("action")
        reset_return = f.reset()
        close_return = f.close()
        assert isinstance(EnvironmentBase, ABCMeta)
        assert start_return is None
        assert step_return is None
        assert reset_return is None
        assert close_return is None
