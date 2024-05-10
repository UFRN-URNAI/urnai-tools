import os
from abc import ABC, abstractmethod


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

    @abstractmethod
    def save(self, savepath):
        """
        This method saves pickle objects
        and extra stuff needed
        """
        ...

    @abstractmethod
    def load(self, loadpath):
        """
        This method loads pickle objects
        and extra stuff needed
        """
        ...

    def _restore_attributes(self, dict_to_restore):
        for key in dict_to_restore:
            if key not in self.attr_block_list and key != 'attr_block_list':
                setattr(self, key, dict_to_restore[key])
