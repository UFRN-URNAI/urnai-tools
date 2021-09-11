class Games:
    DRTS = "deep_rts"
    SC2 = "starcraft_ii"

class Libraries:
    KERAS = "KerasDeepNeuralNetwork"
    PYTORCH = "PyTorchDeepNeuralNetwork"
    TENSORFLOW = "tensorflow"

class RTSGeneralization:

    ACTION_DRTS_DO_NOTHING = 17
    ACTION_DRTS_BUILD_FARM = 18
    ACTION_DRTS_BUILD_BARRACK = 19
    ACTION_DRTS_BUILD_FOOTMAN = 20

    METHOD_SINGLE = "single_environment"
    METHOD_MULTIPLE = "multiple_environment"

    STATE_MAP = "map"
    STATE_NON_SPATIAL = "non_spatial_only"
    STATE_BOTH = "map_and_non_spatial"
    STATE_MAP_DEFAULT_REDUCTIONFACTOR = 1
    STATE_MAXIMUM_X = 64 
    STATE_MAXIMUM_Y = 64 
    STATE_MAX_COLL_DIST = 15 
    STATE_MIN_COLL_ARMY_X_POS = 0 
    STATE_MIN_COLL_ARMY_Y_POS = 0 
    STATE_MAX_COLL_ARMY_X_POS = 64 
    STATE_MAX_COLL_ARMY_Y_POS = 64 
    STATE_MAXIMUM_NUMBER_OF_MINERAL_SHARDS = 20 
    STATE_MAXIMUM_GOLD_OR_MINERALS = 10000 
    MAXIMUM_NUMBER_OF_FARM_OR_SUPPLY_DEPOT = 1
    MAXIMUM_NUMBER_OF_BARRACKS = 1
    MAXIMUM_NUMBER_OF_ARCHERS_MARINES = 20

# geting list of keys in Libraries.__dict__ that do not start with _ to exclude internal atributes
# returns: ['KERAS', 'PYTORCH', 'TENSORFLOW']
libkeys = list(filter( lambda x: not x.startswith("_"), list(Libraries.__dict__.keys()) ))
# geting a list of values correspondent to the keys obtained above in Libraries.__dict__
# returns: ['keras', 'pytorch', 'tensorflow']
listoflibs = [Libraries.__dict__[x] for x in libkeys]
