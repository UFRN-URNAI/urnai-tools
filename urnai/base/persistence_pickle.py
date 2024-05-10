import os
import pickle
import tempfile
from multiprocessing import Process

from urnai.base.persistence import Persistence


class PersistencePickle(Persistence):
    """
    This interface represents the concept of a class that can be saved to disk.
    The heir class should define a constant or attribute as a default filename
	to save on disk.
    """

    def __init__(self, threaded_saving=False):
        super().__init__(threaded_saving)

    def _save(self, persist_path):
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
        path = super().get_full_persistance_path(persist_path)
		
        os.makedirs(os.path.dirname(path), exist_ok=True)
		
        with open(path, 'wb') as pickle_out:
            pickle.dump(self._get_pickleable_dict(), pickle_out)

    def _load(self, persist_path):
        """
        This method loads a list instance
        saved by pickle.
        """
        pickle_path = super().get_full_persistance_path(persist_path)
        exists_pickle = os.path.isfile(pickle_path)
        
        if exists_pickle and os.path.getsize(pickle_path) > 0:
            with open(pickle_path, 'rb') as pickle_in:
                pickle_dict = pickle.load(pickle_in)
                super()._restore_attributes(pickle_dict)

    def _get_attributes(self):
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
                if ("can't pickle" not in str(type_error) or
                 'cannot pickle' not in str(type_error)):
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
                if 'ctypes objects' not in str(value_error):
                    raise
                continue

        return pickleable_list

    def _get_dict(self):
        pickleable_attr_dict = {}

        for attr in self._get_pickleable_attributes():
            pickleable_attr_dict[attr] = getattr(self, attr)

        return pickleable_attr_dict
