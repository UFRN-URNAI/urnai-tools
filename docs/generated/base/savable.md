# File savable.py

## Class SavableAttr

### Method __init__()

- Arguments: value

## Class Savable

This interface represents the concept of a class that can be saved to disk. The heir class should
define a constant or attribute as a default filename to save on disk.

### Method __init__()

- No Arguments.

### Method get_default_save_stamp()

This method returns the default file name that should be used while persisting the object.

- No Arguments.

### Method save_pickle()

This method saves our instance using pickle.

First it checks which attributes should be saved using pickle, the ones which are not are backuped.

Then all unpickleable attributes are set to None and the object is pickled.

Finally the nulled attributes are restored.

- Arguments: persist_path

### Method save_extra()

This method should be implemented when some extra persistence is to be saved.

- Arguments: persist_path

### Method load_pickle()

This method loads a list instance saved by pickle.

- Arguments: persist_path

### Method load_extra()

This method should be implemented when some extra persistence is to be loaded.

- Arguments: persist_path

### Method get_full_persistance_pickle_path()

This method returns the default persistance pickle path.

- Arguments: persist_path

### Method get_full_persistance_tensorflow_path()

This method returns the default persistance tensorflow path.

- Arguments: persist_path

### Method get_full_persistance_path()

This method returns the default persistance path.

- Arguments: persist_path

### Method save()

This method saves pickle objects and extra stuff needed

- Arguments: savepath

### Method load()

This method loads pickle objects and extra stuff needed

- Arguments: loadpath

### Method get_pickleable_attributes()

This method returns a list of pickeable attributes. If you wish to blacklist one particular
pickleable attribute, put it in self.pickle_black_list as a string.

- No Arguments.

### Method get_pickleable_dict()

- No Arguments.

### Method restore_pickleable_attributes()

- Arguments: dict_to_restore
