import unittest
from abc import ABCMeta

from urnai.states.state_base import StateBase


class TestStateBase(unittest.TestCase):

    def test_abstract_methods(self):
        StateBase.__abstractmethods__ = set()

        class FakeState(StateBase):
            def __init__(self):
                super().__init__()

        fake_state = FakeState()
        update_return = fake_state.update("observation")
        state = fake_state.state
        dimension = fake_state.dimension
        reset_return = fake_state.reset()
        assert isinstance(StateBase, ABCMeta)
        assert update_return is None
        assert state is None
        assert dimension is None
        assert reset_return is None
