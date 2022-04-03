import argparse
import importlib
import os
import time

from urnai.runner.base.command import Command
from urnai.utils.reporter import Reporter as rp


class DeepRTSCommand(Command):

    def __init__(self, top_parser, subparser):
        super().__init__(top_parser, subparser)

        self.command = 'drts'
        self.help = 'Subcommand used to call DeepRTS related helpers.'
        self.flags = [
            {
                'command': '--drts-map', 'help': 'Map to work with when using drts command.',
                'type': str,
                'metavar': 'MAP_PATH',
                'action': 'store',
            },
            {
                'command': '--extract-specs',
                'help': 'This will export every map layer to a csv file. Other userful'
                        + ' information will be on a JSON file. This switch works with both'
                        + ' sc2 and drts commands.',
                'action': 'store_true',
            },
            {
                'command': '--drts-map-specs',
                'help': 'Directory to work with when building a drts map.',
                'type': str,
                'metavar': 'DRTS_MAP_SPEC_PATH',
                'action': 'store',
            },
            {
                'command': '--build-map',
                'help': 'This will build a map inside the directory informed with'
                        + ' --drts-map-specs. If you need a template, you should use'
                        + ' --extract-specs first. URNAI will generate the needed files'
                        + ' from an existing DeepRTS map.',
                'action': 'store_true',
            },
            {
                'command': '--install',
                'help': 'Install map on DeepRTS.',
                'action': 'store_true',
            },
            {
                'command': '--uninstall',
                'help': 'Uninstall map on DeepRTS.',
                'action': 'store_true',
            },
            {
                'command': '--show-available-maps',
                'help': 'Show installed maps on DeepRTS.',
                'action': 'store_true',
            },
        ]

        self.add_subcommand()

    def run(self, args):
        DeepRTSEnv = importlib.import_module('urnai.envs.deep_rts.DeepRTSEnv')
        drts_utils = importlib.import_module('urnai.utils.drts_utils')

        if args.show_available_maps:
            drts_utils.show_available_maps()
        elif args.drts_map is not None:
            map_name = os.path.basename(args.drts_map)
            full_map_path = os.path.abspath(map_name)

            if args.install:
                drts_utils.install_map(full_map_path)
            elif args.uninstall:
                drts_utils.uninstall_map(full_map_path)
            elif args.extract_specs:
                if len(os.listdir('.')) > 0:
                    answer = ''
                    while not (answer.lower() == 'y' or answer.lower() == 'n'):
                        answer = input(
                            'Current directory is not empty. Do you wish to continue? [y/n]')

                    if answer.lower() == 'y':
                        rp.report('Extracting {map} features...'.format(map=map_name))
                        drts_utils.extract_specs(map_name)
                else:
                    rp.report('Extracting {map} features...'.format(map=map_name))
                    drts_utils.extract_specs(map_name)

            else:
                rp.report('Starting DeepRTS using map ' + map_name)
                drts = DeepRTSEnv(render=True, map=map_name)
                drts.reset()

                try:
                    while True:
                        drts.reset()
                        drts.step(15)
                        time.sleep(1)
                except KeyboardInterrupt:
                    rp.report('Bye!')
        elif args.build_map:
            if args.drts_map_specs is not None:
                if len(os.listdir(args.drts_map_specs)) > 0:
                    drts_utils.build_map(args.drts_map_specs)
                else:
                    rp.report('DeepRTSMapSpecs directory is empty.')
            else:
                rp.report("--drts-map-specs weren't informed.")
        else:
            raise argparse.ArgumentError(message='--drts-map not informed.')
