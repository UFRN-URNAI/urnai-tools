from urnai.runner.base.runner import Runner
from shutil import copyfile
from urnai.utils.reporter import Reporter as rp 
# from urnai.utils import drts_utils
from urnai.utils import sc2_utils
from urnai.envs.sc2 import SC2Env
# from urnai.envs.deep_rts import DeepRTSEnv 
from pysc2.env import sc2_env
from pysc2.lib import actions
import time
import numpy as np
from absl import flags
import json
import os, sys
import argparse

class DeepRTSRunner(Runner):

    COMMAND = 'drts' 
    OPT_COMMANDS = [
            {'command': '--drts-map', 'help': 'Map to work with when using drts command.', 'type' : str, 'metavar' : 'MAP_PATH', 'action' : 'store'},
            {'command': '--extract-specs', 'help': 'This will export every map layer to a csv file. Other userful information will be on a JSON file. This switch works with both sc2 and drts commands.', 'action' : 'store_true'},
            {'command': '--drts-map-specs', 'help': 'Directory to work with when building a drts map.', 'type' : str, 'metavar' : 'DRTS_MAP_SPEC_PATH', 'action' : 'store'},
            {'command': '--build-map', 'help': 'This will build a map inside the directory informed with --drts-map-specs. If you need a template, you should use --extract-specs first. URNAI will generate the needed files from an existing DeepRTS map.', 'action' : 'store_true'},
            {'command': '--install', 'help': 'Install map on DeepRTS.', 'action' : 'store_true'},
            {'command': '--uninstall', 'help': 'Uninstall map on DeepRTS.', 'action' : 'store_true'},
            {'command': '--show-available-maps', 'help': 'Show installed maps on DeepRTS.', 'action' : 'store_true'},
            ]
    
    def __init__(self, parser, args):
        super().__init__(parser, args)

    def run(self):
        if self.args.show_available_maps:
            drts_utils.show_available_maps();
        elif self.args.drts_map is not None:
            map_name = os.path.basename(self.args.drts_map)
            full_map_path = os.path.abspath(map_name)

            if self.args.install:
                drts_utils.install_map(full_map_path)
            elif self.args.uninstall:
                drts_utils.uninstall_map(full_map_path)
            elif self.args.extract_specs:
                if len(os.listdir(".")) > 0:
                    answer = ""
                    while not (answer.lower() == "y" or answer.lower() == "n") :
                        answer = input("Current directory is not empty. Do you wish to continue? [y/n]")

                    if answer.lower() == "y":
                        rp.report("Extracting {map} features...".format(map=map_name))
                        drts_utils.extract_specs(map_name)
                else:
                    rp.report("Extracting {map} features...".format(map=map_name))
                    drts_utils.extract_specs(map_name)
            
            else:
                rp.report("Starting DeepRTS using map " + map_name)
                drts = DeepRTSEnv(render=True,map=map_name)
                drts.reset()

                try:
                    while True:
                        drts.reset()
                        drts.step(15)
                        time.sleep(1)
                except KeyboardInterrupt:
                    rp.report("Bye!")
        elif self.args.build_map:
            if self.args.drts_map_specs is not None:
                if len(os.listdir(self.args.drts_map_specs)) > 0:
                    drts_utils.build_map(self.args.drts_map_specs)
                else:
                    rp.report("DeepRTSMapSpecs directory is empty.")
            else:
                rp.report("--drts-map-specs weren't informed.")
        else:
            raise argparse.ArgumentError(message="--drts-map not informed.")


class TrainerRunner(Runner):

    COMMAND = 'train'
    OPT_COMMANDS = [
            {'command': '--train-file', 'help': 'JSON or CSV solve file, with all the parameters to start the training.', 'type' : str, 'metavar' : 'TRAIN_FILE_PATH', 'action' : 'store'},
#TODO            {'command': '--build-training-file', 'help': 'Helper to build a solve json-file.', 'action' : 'store_true'},
            {'command': '--convert', 'help': 'Training file to convert. Must be used with --output-format option.', 'action' : 'store'},
            {'command': '--output-format', 'help': 'Converted file format . Must be used with --convert option. Accepted values are \'CSV\', \'YAML\' and \'JSON\'.', 'action' : 'store'},
            {'command': '--play', 'help': 'Test agent, without training it, it will ignore train entry on json file.', 'action' : 'store_true'},
            {'command': '--threaded', 'help': 'If there is more than one training in the file, it will be executed in a separated thread.', 'action' : 'store_true'},
            ]

    def __init__(self, parser, args):
        super().__init__(parser, args)

    def run(self):

        if self.args.train_file is not None:
            from urnai.trainers.filetrainer import FileTrainer
            trainer = FileTrainer(self.args.train_file)

            #this is needed in case the environment is starcraft
            for arg in self.args.__dict__.keys():
                arg = "--{}".format(arg.replace("_", "-"))
                if arg in sys.argv: sys.argv.remove(arg)

            trainer.start_training(play_only=self.args.play, threaded_training=self.args.threaded)
        #TODO
        #elif self.args.build_training_file:
        elif self.args.convert is not None:
            if self.args.output_format is not None:
                from urnai.trainers.filetrainer import FileTrainer
                trainer = FileTrainer(self.args.convert)
                trainer.check_trainings()
                output_path = os.path.abspath(os.path.dirname(self.args.convert))+ os.path.sep + os.path.splitext(os.path.basename(self.args.convert))[0] + ".file_format"
                output_text = "{} was converted to {}.".format(
                        os.path.basename(self.args.convert),
                        os.path.basename(output_path)
                        )

                if self.args.output_format == 'CSV':
                    trainer.save_trainings_as_csv(output_path.replace('.file_format', '.csv'))
                    rp.report(output_text.replace('.file_format', '.csv'))
                elif self.args.output_format == 'JSON':
                    trainer.save_trainings_as_json(output_path.replace('.file_format', '.json'))
                    rp.report(output_text.replace('.file_format', '.json'))
                elif self.args.output_format == 'YAML':
                    trainer.save_trainings_as_yaml(output_path.replace('.file_format', '.yaml'))
                    rp.report(output_text.replace('.file_format', '.yaml'))
                else:
                    raise Exception("--out-format must be 'CSV', 'JSON' or 'YAML'.")
            else:
                raise Exception("You must specify --output-format.")
        else:
            raise Exception("You must specify --train-file or --convert (with --output-format set).")

class SC2Runner(Runner):

    COMMAND = 'sc2'
    OPT_COMMANDS = [
            {'command': '--sc2-map', 'help': 'Map to use on Starcraft II.', 'type' : str, 'metavar' : 'MAP_FILE_PATH', 'action' : 'store'},
            ]

    def run (self):
        #remove all args from sys.argv
        #this is needed to run sc2 env without errors
        for arg in self.args.__dict__.keys():
            arg = "--{}".format(arg.replace("_", "-"))
            if arg in sys.argv: sys.argv.remove(arg)


        if self.args.sc2_map is not None:
            if self.args.extract_specs:
                map_name = self.args.sc2_map
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
