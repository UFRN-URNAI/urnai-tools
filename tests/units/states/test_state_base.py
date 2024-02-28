import unittest
from abc import ABCMeta

from urnai.states.state_base import StateBase


class TestStateBase(unittest.TestCase):

    def test_abstract_methods(self):
        StateBase.__abstractmethods__ = set()

        class FakeState(StateBase):
            def __init__(self):
                super().__init__()

        f = FakeState()
        update_return = f.update("observation")
        state = f.get_state
        dimension = f.get_dimension
        reset_return = f.reset()
        assert isinstance(StateBase, ABCMeta)
        assert update_return is None
        assert state is None
        assert dimension is None
        assert reset_return is None
