import unittest
from unittest.mock import patch

from urnai.base.persistence_pickle import PersistencePickle


class FakePersistencePickle(PersistencePickle):
    def __init__(self, threaded_saving=False):
        super().__init__(threaded_saving)

class TestPersistence(unittest.TestCase):

    @patch('urnai.base.persistence_pickle.PersistencePickle._simple_save')
    def test_simple_save(self, mock_simple_save):
        fake_persistence_pickle = FakePersistencePickle()
        persist_path = "test_simple_save"

        mock_simple_save.return_value = "return_value"

        simple_save_return = fake_persistence_pickle._simple_save(persist_path)

        self.assertEqual(simple_save_return, "return_value")

    @patch('urnai.base.persistence_pickle.PersistencePickle.load')
    def test_load(self, mock_load):
        """
        This method creates a FakePersistencePickle with certain values
        and saves it (state1). After that, it changes the object's
        attributes (state2) and loads it back to state1.
        """

        fake_persistence_pickle = FakePersistencePickle()
        persist_path = "test_load"

        mock_load.return_value = "return_value"

        load_return = fake_persistence_pickle.load(persist_path)

        self.assertEqual(load_return, "return_value")

    def test_get_attributes(self):
        fake_persistence_pickle = FakePersistencePickle()

        return_list = fake_persistence_pickle._get_attributes()

        self.assertEqual(return_list, ['threaded_saving'])

    def test_get_dict(self):
        fake_persistence_pickle = FakePersistencePickle()

        return_dict = fake_persistence_pickle._get_dict()

        self.assertEqual(return_dict, {"threaded_saving": False})
