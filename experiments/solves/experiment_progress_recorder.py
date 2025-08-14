class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            print("\nCALL INFO: class is ", cls, " and current instances are ", SingletonMeta._instances, "\n")
            cls._instances[cls] = instance
        return cls._instances[cls]

class ExperimentProgressRecorder(metaclass=SingletonMeta):

    def __init__(self):
        self.progress = 0

    def get_progress(self):
        print("\nGET INFO: progress is ", self.progress, " and id is ", id(self), "\n")
        return self.progress
    
    def set_progress(self, progress):
        self.progress = progress
        print("\nSET INFO: new progress is ", self.progress, " and id is ", id(self), "\n")
    
recorder = ExperimentProgressRecorder()