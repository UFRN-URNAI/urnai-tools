import unittest
from abc import ABCMeta

from urnai.actions.action_base import ActionBase


class TestActionBase(unittest.TestCase):

    def test_abstract_methods(self):
        ActionBase.__abstractmethods__ = set()

        class FakeAction(ActionBase):
            ...

        f = FakeAction()
        run_return = f.run()
        check_return = f.check("observation")
        is_complete_return = f.is_complete()
        assert isinstance(ActionBase, ABCMeta)
        assert run_return is None
        assert check_return is False
        assert is_complete_return is False
