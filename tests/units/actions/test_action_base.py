import unittest
from abc import ABCMeta

from urnai.actions.action_base import ActionBase


class FakeAction(ActionBase):
    ActionBase.__abstractmethods__ = set()
    __id__ = None
    ...

class TestActionBase(unittest.TestCase):

    def test_abstract_methods(self):

        # GIVEN
        fake_action = FakeAction()

        # WHEN
        run_return = fake_action.run()

        # THEN
        assert fake_action.__id__ is None
        assert isinstance(ActionBase, ABCMeta)
        assert run_return is None
