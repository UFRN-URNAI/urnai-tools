class ActionError(Exception):
    pass

class DeprecatedCodeException(Exception):
    pass

class CommandsNotUniqueError(Exception):
    pass

class EnvironmentNotSupportedError(Exception):
    pass

class WadNotFoundError(Exception):
    pass

class IncoherentBuildModelError(Exception):
    pass

class UnsupportedBuildModelLayerTypeError(Exception):
    pass

class UnsupportedVizDoomRes(Exception):
    pass

class ClassNotFoundError(Exception):
    pass

class NoEnemyArmyError(Exception):
    pass

class IncorrectDeepRTSMapDataError(Exception):
    pass

class DeepRTSEnvError(Exception):
    pass

class MapNotFoundError(Exception):
    pass

class UnsupportedTrainingMethodError(Exception):
    pass

class FileFormatNotSupportedError(Exception):
    pass

class UnsuportedLibraryError(Exception):
    def __init__(self, lib):
        self.lib = lib
        self.message = "\'" + str(self.lib) + "\' is not a supported Machine Learning Library, check for typing errors."
        super().__init__(self.message)

class DifferentModelSizeError(Exception):
    pass
