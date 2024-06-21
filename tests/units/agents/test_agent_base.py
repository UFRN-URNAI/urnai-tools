import unittest
from abc import ABCMeta

from urnai.agents.agent_base import AgentBase
from urnai.models.model_base import ModelBase
from urnai.states.state_base import StateBase


class FakeAgent(AgentBase):
    AgentBase.__abstractmethods__ = set()

    def __init__(self, action_space, 
                 state_space : StateBase, 
                 model : ModelBase, 
                 reward_builder):
        super().__init__(action_space, state_space, model, reward_builder)
    ...

class TestAgentBase(unittest.TestCase):

    def test_abstract_methods(self):

        # GIVEN
        fake_agent = FakeAgent(None, None, None, None)

        # WHEN
        step_return = fake_agent.step()

        # THEN
        assert isinstance(AgentBase, ABCMeta)
        assert step_return is None
