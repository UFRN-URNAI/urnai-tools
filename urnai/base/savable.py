import os
import pickle
import tempfile
from multiprocessing import Process

#import time
#from urnai.utils.reporter import Reporter as rp


class SavableAttr:
    def __init__(self, value):
        self.value = value


class Savable:
    """
    This interface represents the concept of a class that can be saved to disk.
    The heir class should define a constant or attribute as a default filename
	to save on disk.
    """

    def __init__(self, threaded_saving=False):
        self.threaded_saving = threaded_saving
        self.pickle_black_list = []

    def get_default_save_stamp(self):
        """
        This method returns the default
        file name that should be used while
        persisting the object.
        """
        return self.__class__.__name__ + '_'

    def save_pickle(self, persist_path):
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
        path = self.get_full_persistance_pickle_path(persist_path)
		
		# creating necessary directories for the file
        os.makedirs(os.path.dirname(path), exist_ok=True)
		
        with open(path, 'wb') as pickle_out:
            pickle.dump(self.get_pickleable_dict(), pickle_out)

    def save_extra(self, persist_path):
        """
        This method should be implemented when
        some extra persistence is to be saved.
        """
        pass

    def load_pickle(self, persist_path):
        """
        This method loads a list instance
        saved by pickle.
        """
        # Check if pickle file exists
        pickle_path = self.get_full_persistance_pickle_path(persist_path)
        exists_pickle = os.path.isfile(pickle_path)
        
        # If yes, load it
        if exists_pickle and os.path.getsize(pickle_path) > 0:
            with open(pickle_path, 'rb') as pickle_in:
                pickle_dict = pickle.load(pickle_in)
                self.restore_pickleable_attributes(pickle_dict)
                '''
                rp.report(
                    '**************************************** \n Loaded pickle for '
                    + self.get_default_save_stamp()
                    + ' loaded. \n****************************************',
                2)
				'''

    def load_extra(self, persist_path):
        """
        This method should be implemented when
        some extra persistence is to be loaded.
        """
        pass

    def get_full_persistance_pickle_path(self, persist_path):
        """This method returns the default persistance pickle path."""
        return (persist_path + os.path.sep + self.get_default_save_stamp()
		+ '.pkl')

    def get_full_persistance_tensorflow_path(self, persist_path):
        """This method returns the default persistance tensorflow path."""
        return (persist_path + os.path.sep + self.get_default_save_stamp()
		+ 'tensorflow_')

    def get_full_persistance_path(self, persist_path):
        """This method returns the default persistance path."""
        return persist_path + os.path.sep + self.get_default_save_stamp()

    def get_full_persistance_pytorch_path(self, persist_path):
        """This method returns the default persistance pytorch path."""
        return (persist_path + os.path.sep + self.get_default_save_stamp()
		+ '.pt')

    def save(self, savepath):
        """
        This method saves pickle objects
        and extra stuff needed
        """
        #rp.report('Saving {} object...'.format
		#(self.__class__.__name__), verbosity_lvl=1)
        #start_time = time.time()
        
        if not self.threaded_saving:
            self.save_pickle(savepath)
        else:
            self.threaded_pickle_save(savepath)

        self.save_extra(savepath)
        
        #end_time = time.time()
        '''
        rp.report('It took {} seconds to save {} object!'
		.format(end_time - start_time, self.__class__.__name__),
        verbosity_lvl=2)
        '''

    def threaded_pickle_save(self, savepath):
        """
        This method saves pickleable
        stuff in a separate thread
        """
        #rp.report('THREADED Saving {} object...'
		#.format(self.__class__.__name__), verbosity_lvl=1)
		
        p = Process(target=self.save_pickle, args=(savepath,))
        p.start()

    def load(self, loadpath):
        """
        This method loads pickle objects
        and extra stuff needed
        """
        self.load_pickle(loadpath)
        self.load_extra(loadpath)

    def get_pickleable_attributes(self):
        """
        This method returns a list of
        pickeable attributes. If you wish
        to blacklist one particular
        pickleable attribute, put it
        in self.pickle_black_list as a
        string.
        """
        if not hasattr(self, 'pickle_black_list') or self.pickle_black_list is None:
            self.pickle_black_list = []

        full_attr_list = [a for a in dir(self) if not a.startswith('__')
                          and not callable(getattr(self, a))
                          and a not in self.pickle_black_list
                          and a != 'pickle_black_list']
        pickleable_list = []

        for attr in full_attr_list:
            try:
                with tempfile.NamedTemporaryFile() as tmp_file:
                    pickle.dump(getattr(self, attr), tmp_file)
                    tmp_file.flush()
                
                pickleable_list.append(attr)
                
            except pickle.PicklingError:
                continue
            
            except TypeError as te:
                if "can't pickle" in str(te):
                    continue
                if 'cannot pickle' in str(te):
                    continue
                else:
                    raise
            
            except NotImplementedError as nie:
                if (str(nie) == 
                'numpy() is only available when eager execution is enabled.'):
                    continue
                else:
                    raise
            
            except AttributeError as ae:
                if ("Can't pickle" in str(ae) or 
                "object has no attribute '__getstate__'"):
                    continue
            
            except ValueError as ve:
                if 'ctypes objects' in str(ve):
                    continue
                else:
                    raise

        return pickleable_list

    def get_pickleable_dict(self):
        pickleable_attr_dict = {}

        for attr in self.get_pickleable_attributes():
            pickleable_attr_dict[attr] = getattr(self, attr)

        return pickleable_attr_dict

    def restore_pickleable_attributes(self, dict_to_restore):
        for attr in dict_to_restore:
            if hasattr(self, 'pickle_black_list'):
                if attr not in self.pickle_black_list and attr != 'pickle_black_list':
                    setattr(self, attr, dict_to_restore[attr])
            else:
                setattr(self, attr, dict_to_restore[attr])
