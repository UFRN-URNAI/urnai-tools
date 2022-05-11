class Games:
    DRTS = 'deep_rts'
    SC2 = 'starcraft_ii'


class Libraries:
    KERAS = 'keras'
    PYTORCH = 'pytorch'
    TENSORFLOW = 'tensorflow'
    KERAS_E_TRACES = 'keras_e_traces'


class RTSGeneralization:
    ACTION_DRTS_DO_NOTHING = 17
    ACTION_DRTS_BUILD_FARM = 18
    ACTION_DRTS_BUILD_BARRACK = 19
    ACTION_DRTS_BUILD_FOOTMAN = 20

    METHOD_SINGLE = 'single_environment'
    METHOD_MULTIPLE = 'multiple_environment'

    STATE_MAP = 'map'
    STATE_NON_SPATIAL = 'non_spatial_only'
    STATE_BOTH = 'map_and_non_spatial'
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

class Trainings:
    DEFAULT_TRAINING = {
        "env" : {
            "class" : "GymEnv",
            "params" : {
                "id" : "CartPole-v1"
            }
        },
        "action_wrapper" : {
            "class" : "GymWrapper",
            "params" : {
                "gym_env_actions" : 2
            } 
        },
        "state_builder" : {
            "class" : "PureState",
            "params" : {
                "state_dim" : 4 
            }
        },
        "model" : {
            "class" : "DoubleDeepQLearning",
            "params" : {
                "learning_rate" : 0.001,
                "gamma" : 0.99,
                "epsilon_decay" : 0.9997, 
                "epsilon_min" : 0.005,
                "memory_maxlen" : 50000,
                "min_memory_size" : 1000,
                "build_model" : [
                    {
                        "type": "input", 
                        "nodes": 25, 
                        "shape": []
                    }, 
                    {
                        "type": "fullyconn",
                        "nodes": 25,
                        "name": "default0"
                    },
                    {
                        "type": "output", 
                        "length": 2
                    }
                ]
             }
        },
        "reward" : {
            "class" : "PureReward",
            "params" : {} 
        },
        "agent" : {
            "class" : "GenericAgent",
            "params" : {} 
        },
        "trainer" : {
            "class" : "Trainer",
            "params" : {
                "file_name" : "cartpole-v1_ddql_25x25_json",
                "save_every" : 100,
                "enable_save" : True,
                "max_training_episodes" : 1000,
                "max_steps_training" : 520,
                "max_test_episodes" : 100,
                "max_steps_testing" : 520,
                "rolling_avg_window_size" : 50
            }
        }
    }

    DUMMY_TRAINING = {
        "env" : {
            "class" : "ExampleEnv",
            "params" : {
                "example_param" : "example_value",
            }
        },
        "action_wrapper" : {
            "class" : "ExampleActionWrapper",
            "params" : {
                "example_param" : "example_value",
            } 
        },
        "state_builder" : {
            "class" : "ExampleStateBuilder",
            "params" : {
                "example_param" : "example_value",
            }
        },
        "model" : {
            "class" : "ExampleModel",
            "params" : {
                "example_param" : "example_value",
                "build_model" : [
                    {
                        "type": "input", 
                        "nodes": 25, 
                        "shape": []
                    }, 
                    {
                        "type": "fullyconn",
                        "nodes": 25,
                        "name": "default0"
                    },
                    {
                        "type": "output", 
                        "length": 2
                    }
                ]
             }
        },
        "reward" : {
            "class" : "PureReward",
            "params" : {} 
        },
        "agent" : {
            "class" : "GenericAgent",
            "params" : {} 
        },
        "trainer" : {
            "class" : "Trainer",
            "params" : {
                "file_name" : "example",
                "save_every" : 0,
                "enable_save" : True,
                "max_training_episodes" : 0,
                "max_steps_training" : 0,
                "max_test_episodes" : 0,
                "max_steps_testing" : 0,
            }
        }
    }


# geting list of keys in Libraries.__dict__ that do not start with _ to exclude internal atributes
# returns: ['KERAS', 'PYTORCH', 'TENSORFLOW']
libkeys = list(filter(lambda x: not x.startswith('_'), list(Libraries.__dict__.keys())))
# geting a list of values correspondent to the keys obtained above in Libraries.__dict__
# returns: ['keras', 'pytorch', 'tensorflow']
listoflibs = [Libraries.__dict__[x] for x in libkeys]
