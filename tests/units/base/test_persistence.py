import os
import unittest
from abc import ABCMeta

from urnai.base.persistence import Persistence


class FakePersistence(Persistence):
    Persistence.__abstractmethods__ = set()

    def __init__(self, threaded_saving=False):
        super().__init__(threaded_saving)

class TestPersistence(unittest.TestCase):

    def test_abstract_methods(self):

        fake_persistence = FakePersistence()
        _simple_save_return = fake_persistence._simple_save(".")
        load_return = fake_persistence.load(".")
        _get_dict_return = fake_persistence._get_dict()
        _get_attributes_return = fake_persistence._get_attributes()

        assert isinstance(Persistence, ABCMeta)
        assert _simple_save_return is None
        assert load_return is None
        assert _get_dict_return is None
        assert _get_attributes_return is None

    def test_get_default_save_stamp(self):
        fake_persistence = FakePersistence()
        self.assertEqual(fake_persistence._get_default_save_stamp(),
                          fake_persistence.__class__.__name__ + '_')

    def test_get_full_persistance_path(self):
        fake_persistence = FakePersistence()
        persist_path = "test"
        self.assertEqual(
            fake_persistence.get_full_persistance_path(persist_path),
            persist_path + os.path.sep + 
            fake_persistence._get_default_save_stamp())
        
    def test_save(self):
        fake_persistence = FakePersistence()
        persist_path = "test"

        save_return = fake_persistence.save(persist_path)
        assert save_return is None

        fake_persistence_threaded = FakePersistence(threaded_saving=True)
        persist_path = "test"

        fake_persistence_threaded.save(persist_path)

        for process in fake_persistence_threaded.processes:
            assert process.is_alive() is True

    def test_threaded_save(self):
        fake_persistence = FakePersistence(threaded_saving=True)
        persist_path = "test"

        fake_persistence.save(persist_path)

        for process in fake_persistence.processes:
            assert process.is_alive() is True

    def test_restore_attributes(self):
        fake_persistence = FakePersistence()
        dict_to_restore = {"TestAttribute1": 314, "TestAttribute2": "string"}
        fake_persistence.attr_block_list = ["TestAttribute2"]
        fake_persistence._restore_attributes(dict_to_restore)

        assert hasattr(fake_persistence, "TestAttribute1") is True
        assert hasattr(fake_persistence, "TestAttribute2") is False
        self.assertEqual(fake_persistence.TestAttribute1, 314)
