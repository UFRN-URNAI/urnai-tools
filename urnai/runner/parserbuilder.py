import argparse

from urnai.runner.commands.drts import DeepRTSCommand
from urnai.runner.commands.sc2 import SC2Command
from urnai.runner.commands.trainer import TrainerCommand


class ParserBuilder:
    DESCRIPTION = 'A modular Deep Reinforcement Learning library that supports multiple' \
                  + ' environments, such as PySC2, OpenAI Gym, and PyGame Learning Environment.'
    COMMANDS = [DeepRTSCommand, SC2Command, TrainerCommand]

    @staticmethod
    def DefaultParser():
        parser = argparse.ArgumentParser(description=ParserBuilder.DESCRIPTION)
        subparser = parser.add_subparsers(title='subcommands',
                                          description='valid subcommands',
                                          help='additional help')

        for cls in ParserBuilder.COMMANDS:
            cls(parser, subparser)

        return parser
