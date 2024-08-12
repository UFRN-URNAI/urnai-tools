import os
import pickle
import tempfile

from urnai.base.persistence import Persistence


class PersistencePickle(Persistence):
    """
    This interface represents the concept of a class that can be saved to disk.
    The heir class should define a constant or attribute as a default filename
	to save on disk.
    """

    def __init__(self, object_to_save, threaded_saving=False):
        super().__init__(object_to_save, threaded_saving)

    def _simple_save(self, persist_path):
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
            pickle.dump(self._get_dict(), pickle_out)

    def load(self, persist_path):
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

    def _get_attributes(self):
        """
        This method returns a list of pickeable attributes.
		If you wish to block one particular pickleable attribute, put it
        in self.attr_block_list as a string.
        """
        if not hasattr(self.object_to_save, 'attr_block_list') or self.attr_block_list is None:
            self.attr_block_list = []

        attr_block_list = self.attr_block_list + ['attr_block_list', 'processes']

        full_attr_list = [attr for attr in dir(self.object_to_save) if not attr.startswith('__')
                          and not callable(getattr(self.object_to_save, attr))
                          and attr not in attr_block_list
                          and 'abc' not in attr]
        
        pickleable_list = []

        for key in full_attr_list:
            try:
                with tempfile.NamedTemporaryFile() as tmp_file:
                    pickle.dump(getattr(self.object_to_save, key), tmp_file)
                    tmp_file.flush()
                
                pickleable_list.append(key)
                
            except pickle.PicklingError:
                continue
            
            except TypeError as type_error:
                if ("can't pickle" not in str(type_error) and
                 'cannot pickle' not in str(type_error)):
                    raise TypeError() from type_error
                continue
            
            except NotImplementedError as notimpl_error:
                if (str(notimpl_error) != 
                'numpy() is only available when eager execution is enabled.'):
                    raise NotImplementedError() from notimpl_error
                continue
            
            except AttributeError as attr_error:
                if ("Can't pickle" not in str(attr_error) and 
                "object has no attribute '__getstate__'" not in
                str(attr_error)):
                    raise AttributeError() from attr_error
                continue
            
            except ValueError as value_error:
                if 'ctypes objects' not in str(value_error):
                    raise ValueError() from value_error
                continue

        return pickleable_list

    def _get_dict(self):
        pickleable_attr_dict = {}

        for attr in self._get_attributes():
            pickleable_attr_dict[attr] = getattr(self.object_to_save, attr)

        return pickleable_attr_dict
