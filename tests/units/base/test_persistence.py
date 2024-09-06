import os
import unittest
from abc import ABCMeta

from urnai.base.persistence import Persistence


class FakePersistence(Persistence):
    Persistence.__abstractmethods__ = set()

    def __init__(self, threaded_saving=False):
        super().__init__(threaded_saving)

class SimpleClass:
    def __init__(self):
        pass

class TestPersistence(unittest.TestCase):

    def test_abstract_methods(self):

        # GIVEN
        fake_persistence = FakePersistence()

        # WHEN
        _simple_save_return = fake_persistence._simple_save(".")
        load_return = fake_persistence.load(".")
        _get_dict_return = fake_persistence._get_dict()
        _get_attributes_return = fake_persistence._get_attributes()

        # THEN
        assert isinstance(Persistence, ABCMeta)
        assert _simple_save_return is None
        assert load_return is None
        assert _get_dict_return is None
        assert _get_attributes_return is None

    def test_get_default_save_stamp(self):

        # GIVEN
        fake_persistence = FakePersistence()

        # WHEN
        _get_def_save_stamp_return = fake_persistence._get_default_save_stamp()

        # THEN
        self.assertEqual(_get_def_save_stamp_return,
                          fake_persistence.object_to_save.__class__.__name__ + '_')

    def test_get_full_persistance_path(self):

        # GIVEN
        fake_persistence = FakePersistence()
        persist_path = "test"
        
        # WHEN
        get_full_pers_path_return = fake_persistence.get_full_persistance_path(
            persist_path)

        # THEN
        self.assertEqual(get_full_pers_path_return,
            persist_path + os.path.sep + 
            fake_persistence._get_default_save_stamp())
        
    def test_save(self):

        # GIVEN
        fake_persistence = FakePersistence()
        persist_path = "test"

        # WHEN
        save_return = fake_persistence.save(persist_path)
        fake_persistence_threaded = FakePersistence(threaded_saving=True)
        fake_persistence_threaded.save(persist_path)

        # THEN
        assert save_return is None
        for process in fake_persistence_threaded.processes:
            assert process.is_alive() is True

    def test_threaded_save(self):
        
        # GIVEN
        fake_persistence = FakePersistence(threaded_saving=True)
        persist_path = "test"

        # WHEN
        fake_persistence.save(persist_path)

        # THEN
        for process in fake_persistence.processes:
            assert process.is_alive() is True

    def test_restore_attributes(self):

        # GIVEN
        obj_to_save = SimpleClass()
        fake_persistence = FakePersistence(obj_to_save)
        dict_to_restore = {"TestAttribute1": 314, "TestAttribute2": "string"}

        # WHEN
        fake_persistence.attr_block_list = ["TestAttribute2"]
        fake_persistence._restore_attributes(dict_to_restore)

        # THEN
        assert hasattr(fake_persistence.object_to_save, "TestAttribute1") is True
        assert hasattr(fake_persistence.object_to_save, "TestAttribute2") is False
        self.assertEqual(fake_persistence.object_to_save.TestAttribute1, 314)
