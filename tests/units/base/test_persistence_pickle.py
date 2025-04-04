import unittest
from unittest.mock import mock_open, patch

from urnai.base.persistence_pickle import PersistencePickle


class TestPersistence(unittest.TestCase):
    @patch('urnai.base.persistence_pickle.os.makedirs')
    @patch('urnai.base.persistence_pickle.open', mock_open(read_data=""))
    @patch('urnai.base.persistence_pickle.pickle.dump')
    def test_simple_save(self, mock_pickle_dump, mock_makedirs):
        # GIVEN
        persistence_pickle = PersistencePickle()
        persist_path = "test_simple_save"
        mock_makedirs.return_value = ""
        mock_pickle_dump.return_value = ""

        # WHEN
        persistence_pickle._simple_save(persist_path)

        # THEN
        mock_makedirs.assert_called_once_with(persist_path, exist_ok=True)
        self.assertEqual(mock_pickle_dump.call_count, 2)

    @patch('urnai.base.persistence_pickle.PersistencePickle.load')
    def test_load(self, mock_load):
        """
        This method creates a FakePersistencePickle with certain values
        and saves it (state1). After that, it changes the object's
        attributes (state2) and loads it back to state1.
        """
        # GIVEN
        persistence_pickle = PersistencePickle()
        persist_path = "test_load"
        mock_load.return_value = "return_value"

        # WHEN
        load_return = persistence_pickle.load(persist_path)

        # THEN
        self.assertEqual(load_return, "return_value")

    def test_get_attributes(self):
        # GIVEN
        persistence_pickle = PersistencePickle()

        # WHEN
        return_list = persistence_pickle._get_attributes()

        # THEN
        self.assertEqual(return_list, ['threaded_saving'])

    def test_get_dict(self):
        # GIVEN
        persistence_pickle = PersistencePickle()

        # WHEN
        return_dict = persistence_pickle._get_dict()

        # THEN
        self.assertEqual(return_dict, {"threaded_saving": False})
