import os
import pickle
import tempfile
from multiprocessing import Process


class Savable:
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

    def _save_pickle(self, persist_path):
        """
        This method saves our instance
        using pickle.

        First it checks which attributes should be
        saved using pickle, the ones which are not
        are backuped.

        Then all unpickleable attributes are set to None
        and the object is pickled.

        Finally the nulled attributes are
        restored.
        """
        path = self.get_full_persistance_path(persist_path)
		
        os.makedirs(os.path.dirname(path), exist_ok=True)
		
        with open(path, 'wb') as pickle_out:
            pickle.dump(self._get_pickleable_dict(), pickle_out)

    def _save_extra(self, persist_path):
        """
        This method should be implemented when
        some extra persistence is to be saved.
        """
        pass

    def _load_pickle(self, persist_path):
        """
        This method loads a list instance
        saved by pickle.
        """
        pickle_path = self.get_full_persistance_path(persist_path)
        exists_pickle = os.path.isfile(pickle_path)
        
        if exists_pickle and os.path.getsize(pickle_path) > 0:
            with open(pickle_path, 'rb') as pickle_in:
                pickle_dict = pickle.load(pickle_in)
                self._restore_attributes(pickle_dict)

    def _load_extra(self, persist_path):
        """
        This method should be implemented when
        some extra persistence is to be loaded.
        """
        pass

    def get_full_persistance_path(self, persist_path):
        """This method returns the default persistance path."""
        return persist_path + os.path.sep + self._get_default_save_stamp()

    def save(self, savepath):
        """
        This method saves pickle objects
        and extra stuff needed
        """
        
        if not self.threaded_saving:
            self._save_pickle(savepath)
        else:
            self._threaded_pickle_save(savepath)

        self._save_extra(savepath)

    def _threaded_pickle_save(self, savepath):
        """
        This method saves pickleable
        stuff in a separate thread
        """
		
        new_process = Process(target=self._save_pickle, args=(savepath,))
        new_process.start()

    def load(self, loadpath):
        """
        This method loads pickle objects
        and extra stuff needed
        """
        self._load_pickle(loadpath)
        self._load_extra(loadpath)

    def _get_pickleable_attributes(self):
        """
        This method returns a list of pickeable attributes.
		If you wish to block one particular pickleable attribute, put it
        in self.attr_block_list as a string.
        """
        if not hasattr(self, 'attr_block_list') or self.attr_block_list is None:
            self.attr_block_list = []

        full_attr_list = [attr for attr in dir(self) if not attr.startswith('__')
                          and not callable(getattr(self, attr))
                          and attr not in self.attr_block_list
                          and attr != 'attr_block_list']
        pickleable_list = []

        for key in full_attr_list:
            try:
                with tempfile.NamedTemporaryFile() as tmp_file:
                    pickle.dump(getattr(self, key), tmp_file)
                    tmp_file.flush()
                
                pickleable_list.append(key)
                
            except pickle.PicklingError:
                continue
            
            except TypeError as type_error:
                if (not "can't pickle" in str(type_error)) or (not 'cannot pickle' in str(type_error)):
                    raise
                continue
            
            except NotImplementedError as notimpl_error:
                if (str(notimpl_error) != 
                'numpy() is only available when eager execution is enabled.'):
                    raise
                continue
            
            except AttributeError as attr_error:
                if ("Can't pickle" in str(attr_error) or 
                "object has no attribute '__getstate__'"):
                    continue
            
            except ValueError as value_error:
                if not 'ctypes objects' in str(value_error):
                    raise
                continue

        return pickleable_list

    def _get_pickleable_dict(self):
        pickleable_attr_dict = {}

        for attr in self._get_pickleable_attributes():
            pickleable_attr_dict[attr] = getattr(self, attr)

        return pickleable_attr_dict

    def _restore_attributes(self, dict_to_restore):
        for key in dict_to_restore:
            if key not in self.attr_block_list and key != 'attr_block_list':
                setattr(self, key, dict_to_restore[key])
