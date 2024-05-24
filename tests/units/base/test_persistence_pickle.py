import os
import unittest

from urnai.base.persistence_pickle import PersistencePickle


class FakePersistencePickle(PersistencePickle):
    def __init__(self, threaded_saving=False):
        super().__init__(threaded_saving)

class TestPersistence(unittest.TestCase):

    def test_simple_save(self):
        fake_persistence_pickle = FakePersistencePickle()
        persist_path = "test_simple_save"

        fake_persistence_pickle._simple_save(persist_path)

        assert os.path.exists(persist_path) is True

        #os.remove(persist_path) <- permission denied !!!

    def test_load(self):
        """
        This method creates a FakePersistencePickle with certain values
        and saves it (state1). After that, it changes the object's
        attributes (state2) and loads it back to state1.
        """

        fake_persistence_pickle = FakePersistencePickle()
        persist_path = "test_load"

        fake_persistence_pickle.threaded_saving = False
        fake_persistence_pickle.attr_block_list = ['state1_test']
        fake_persistence_pickle.processes = ['state1_test2']

        fake_persistence_pickle.save(persist_path)

        fake_persistence_pickle.threaded_saving = True
        fake_persistence_pickle.attr_block_list = ['state2_test']
        fake_persistence_pickle.processes = ['state2_test2']

        fake_persistence_pickle.load(persist_path)
            
        self.assertEqual(fake_persistence_pickle.threaded_saving, False)
        self.assertEqual(fake_persistence_pickle.attr_block_list, ['state2_test'])
        self.assertEqual(fake_persistence_pickle.processes, ['state2_test2'])
		
        #os.remove(persist_path) <- permission denied !!!

    def test_get_attributes(self):
        fake_persistence_pickle = FakePersistencePickle()

        return_list = fake_persistence_pickle._get_attributes()

        self.assertEqual(return_list, ['threaded_saving'])

    def test_get_dict(self):
        fake_persistence_pickle = FakePersistencePickle()

        return_dict = fake_persistence_pickle._get_dict()

        self.assertEqual(return_dict, {"threaded_saving": False})
