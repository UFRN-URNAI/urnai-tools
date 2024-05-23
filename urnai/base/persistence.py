import os
from abc import ABC, abstractmethod
from multiprocessing import Process


class Persistence(ABC):
    """
    This interface represents the concept of a class that can be saved to disk.
    The heir class should define a constant or attribute as a default filename
	to save on disk.
    """

    def __init__(self, threaded_saving=False):
        self.threaded_saving = threaded_saving
        self.attr_block_list = []

    def _get_default_save_stamp(self):
        """
        This method returns the default
        file name that should be used while
        persisting the object.
        """
        return self.__class__.__name__ + '_'

    def get_full_persistance_path(self, persist_path):
        """This method returns the default persistance path."""
        return persist_path + os.path.sep + self._get_default_save_stamp()

    def save(self, savepath):
        if self.threaded_saving:
            self._threaded_save(savepath)
        else:
            self._simple_save(savepath)

    def _threaded_save(self, savepath):
        """
        This method saves pickleable
        elements in a separate thread
        """
        new_process = Process(target=self._simple_save, args=(savepath,))
        new_process.start()

    @abstractmethod
    def _simple_save(self, savepath):
        """
        This method handles logic related
        to non-threaded saving in the child class
        """
        ...

    @abstractmethod
    def load(self, savepath):
        ...

    @abstractmethod
    def _get_dict(self):
        """
        This method returns a dict with
        all the attributes that will be
        saved
        """
        ...

    @abstractmethod
    def _get_attributes(self):
        """
        This method returns all of the
        attribute names that can be saved
        except those in blocklist
        """
        ...

    def _restore_attributes(self, dict_to_restore):
        for key in dict_to_restore:
            if key not in self.attr_block_list and key != 'attr_block_list':
                setattr(self, key, dict_to_restore[key])
