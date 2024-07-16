import unittest
from abc import ABCMeta

from urnai.actions.action_space_base import ActionSpaceBase
from urnai.agents.agent_base import AgentBase
from urnai.models.model_base import ModelBase
from urnai.states.state_base import StateBase


class FakeAgent(AgentBase):
    AgentBase.__abstractmethods__ = set()

    def __init__(self, action_space,
                 state_space : StateBase,
                 model : ModelBase,
                 reward):
        super().__init__(action_space, state_space, model, reward)

class FakeState(StateBase):
    StateBase.__abstractmethods__ = set()

class FakeModel(ModelBase):
    ModelBase.__abstractmethods__ = set()

    def ep_reset(self, ep):
        return None

class FakeReward:
    def get_reward(self, obs, reward, done):
        return None
    def reset(self):
        return None

class FakeActionSpace(ActionSpaceBase):
    ActionSpaceBase.__abstractmethods__ = set()

class TestAgentBase(unittest.TestCase):

    def test_abstract_methods(self):
        # GIVEN
        fake_agent = FakeAgent(None, None, None, None)

        # WHEN
        step_return = fake_agent.step()
        choose_action_return = fake_agent.choose_action(FakeActionSpace())

        # THEN
        assert isinstance(AgentBase, ABCMeta)
        assert step_return is None
        assert choose_action_return is None

    def test_reset(self):
        # GIVEN
        fake_action_space = FakeActionSpace()
        fake_state_space = FakeState()
        fake_model = FakeModel()
        fake_reward = FakeReward()
        fake_agent = FakeAgent(
            fake_action_space,
            fake_state_space,
            fake_model,
            fake_reward
        )

        # WHEN
        reset_return = fake_agent.reset()

        # THEN
        assert reset_return is None
        assert fake_agent.previous_action is None
        assert fake_agent.previous_state is None

    def test_learn(self):
        # GIVEN
        fake_model = FakeModel()
        fake_agent = FakeAgent(None, None, fake_model, None)

        # WHEN
        learn_return = fake_agent.learn("obs", "reward", "done")

        # THEN
        assert learn_return is None
