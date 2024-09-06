import unittest
from abc import ABCMeta

from urnai.models.model_base import ModelBase


class TestModelBase(unittest.TestCase):

	def test_abstract_methods(self):
		ModelBase.__abstractmethods__ = set()

		class FakeModel(ModelBase):
			def __init__(self):
				super().__init__()

		fake_model = FakeModel()
		learn_return = fake_model.learn("current_state",
			"action", "reward", "next_state")
		predict_return = fake_model.predict("state")
		learning_dict = fake_model.learning_data

		assert isinstance(ModelBase, ABCMeta)
		assert learn_return is None
		assert predict_return is None
		self.assertEqual(learning_dict, {})
