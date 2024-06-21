import unittest
from abc import ABCMeta
from unittest.mock import MagicMock

from urnai.actions.action_space_base import ActionSpaceBase


class FakeActionSpace(ActionSpaceBase):
    ActionSpaceBase.__abstractmethods__ = set()
    ...

class TestActionSpaceBase(unittest.TestCase):

    def test_abstract_methods(self):

        # GIVEN
        fake_action_space = FakeActionSpace()

        # WHEN
        is_action_done_return = fake_action_space.is_action_done()
        reset_return = fake_action_space.reset()
        get_actions_return = fake_action_space.get_actions()
        get_excluded_actions_return = fake_action_space.get_excluded_actions("obs")
        get_actions_return = fake_action_space.get_action(0, "obs")

        # THEN
        assert isinstance(ActionSpaceBase, ABCMeta)
        assert is_action_done_return is None
        assert reset_return is None
        assert get_actions_return is None
        assert get_excluded_actions_return is None

    def test_get_named_actions(self):

        # GIVEN
        fake_action_space = FakeActionSpace()

        # WHEN
        fake_action_space.get_named_actions = MagicMock(return_value=[])

        # THEN
        self.assertEqual(fake_action_space.get_named_actions(), [])

    def test_size(self):

        # GIVEN
        fake_action_space = FakeActionSpace()

        # WHEN
        fake_action_space.get_actions = MagicMock(return_value=[])

        # THEN
        self.assertEqual(fake_action_space.size, 0)