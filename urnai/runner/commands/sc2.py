import importlib
import numpy as np
import os
import sys
import json
from absl import flags

from urnai.runner.base.command import Command
from urnai.utils.reporter import Reporter as rp 

class SC2Command(Command):


    def __init__(self, top_parser, subparser):
        super().__init__(top_parser, subparser)
        self.command = 'sc2'
        self.help = 'Subcommand used to call StarCraft II related helpers.'
        self.flags = [
                {'command': '--sc2-map', 'help': 'Map to use on Starcraft II.', 'type' : str, 'metavar' : 'MAP_FILE_PATH', 'action' : 'store'},
                {'command': '--extract-specs', 'help': 'This will export every map layer to a csv file. Other userful information will be on a JSON file. This switch works with both sc2 and drts commands.', 'action' : 'store_true'},
                ]

        self.add_subcommand()

    def run(self, args):
        sc2_utils = importlib.import_module('urnai.utils.sc2_utils')
        SC2Env = importlib.import_module('urnai.envs.sc2.SC2Env')
        sc2_env = importlib.import_module('pysc2.env.sc2_env')
        actions = importlib.import_module('pysc2.lib.actions')
    
    
        
        #remove all args from sys.argv
        #this is needed to run sc2 env without errors
        for arg in args.__dict__.keys():
            arg = "--{}".format(arg.replace("_", "-"))
            if arg in sys.argv: sys.argv.remove(arg)


        if args.sc2_map is not None:
            if args.extract_specs:
                map_name = args.sc2_map
                if len(os.listdir(".")) > 0:
                    answer = ""
                    while not (answer.lower() == "y" or answer.lower() == "n") :
                        answer = input("Current directory is not empty. Do you wish to continue? [y/n]")

                    if answer.lower() == "y":
                        rp.report("Extracting {map} features...".format(map=map_name))
                        self.extract_specs(map_name)
                else:
                    rp.report("Extracting {map} features...".format(map=map_name))
                    self.extract_specs(map_name)
        else:
            rp.report("--sc2-map not informed.")
    
    def get_sc2_env(self, map_name):
        FLAGS = flags.FLAGS
        FLAGS(sys.argv)
        players = [sc2_env.Agent(sc2_env.Race.terran)]
        env = SC2Env(map_name=map_name, render=False, step_mul=32, players=players)
        return env

    def extract_specs(self, map_name):
        #start sc2 env
        env = self.get_sc2_env(map_name) 
        env.start()
        state, reward, done = env.step([actions.RAW_FUNCTIONS.no_op()])

        json_output = {}
        json_output["map_name"] = map_name
        json_output["map_shape"] = state.feature_minimap[0].shape

        with open(map_name +'_info.json', 'w') as outfile:
            outfile.write(json.dumps(json_output, indent=4))

        cont = 0
        for minimap in state.feature_minimap:
            map_csv = np.array(minimap).astype(int) 
            np.savetxt("feature_minimap_layer_{}.csv".format(cont), map_csv, fmt='%i',delimiter=",")
            cont += 1
