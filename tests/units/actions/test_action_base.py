import unittest
from abc import ABCMeta

from urnai.actions.action_base import ActionBase


class TestActionBase(unittest.TestCase):

    def test_abstract_methods(self):
        ActionBase.__abstractmethods__ = set()

        class FakeAction(ActionBase):
            ...

        fake_action = FakeAction()
        run_return = fake_action.run()
        check_return = fake_action.check("observation")
        is_complete_return = fake_action.is_complete()
        assert isinstance(ActionBase, ABCMeta)
        assert run_return is None
        assert check_return is None
        assert is_complete_return is None
