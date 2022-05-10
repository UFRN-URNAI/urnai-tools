from urnai.runner.base.command import Command
from urnai.utils.reporter import Reporter as rp


class GetCommand(Command):

    def __init__(self, top_parser, subparser):
        super().__init__(top_parser, subparser)
        self.command = 'get'
        self.help = 'Subcommand used to get info about Urnai parts.'
        self.flags = [
            {
                'command': '--environments',
                'help': 'List available environments.',
                'action': 'store_true',
            },
            {
                'command': '--environment-params',
                'help': 'Show parameter info about the passed ENVIRONMENT.',
                'type': str, 'metavar': 'ENVIRONMENT', 
                'action': 'store',
            },
            {
                'command': '--agents',
                'help': 'List available agents.',
                'action': 'store_true',
            },
            {
                'command': '--agent-params',
                'help': 'Show parameter info about the passed AGENT.',
                'type': str, 'metavar': 'AGENT', 
                'action': 'store',
            },
            {
                'command': '--action-wrappers',
                'help': 'List available action-wrappers.',
                'action': 'store_true',
            },
            {
                'command': '--action-wrapper-params',
                'help': 'Show parameter info about the passed ACTION_WRAPPER.',
                'type': str, 'metavar': 'ACTION_WRAPPER', 
                'action': 'store',
            },
            {
                'command': '--state-builders',
                'help': 'List available state-builders.',
                'action': 'store_true',
            },
            {
                'command': '--state-builder-params',
                'help': 'Show parameter info about the passed STATE_BUILDER.',
                'type': str, 'metavar': 'STATE_BUILDER', 
                'action': 'store',
            },
            {
                'command': '--rewards',
                'help': 'List available state-builders.',
                'action': 'store_true',
            },
            {
                'command': '--reward-params',
                'help': 'Show parameter info about the passed REWARD.',
                'type': str, 'metavar': 'REWARD', 
                'action': 'store',
            },
            {
                'command': '--models',
                'help': 'List available models.',
                'action': 'store_true',
            },
            {
                'command': '--model-params',
                'help': 'Show parameter info about the passed MODEL.',
                'type': str, 'metavar': 'MODEL', 
                'action': 'store',
            },
            {
                'command': '--default-training-file',
                'help': 'Generate a base training file in YAML format, redirect the output to a file to save or use --save FILE_NAME',
                'action' : 'store_true'
            },
            {
                'command': '--save',
                'help': 'File name to save output to.',
                'type': str, 'metavar': 'FILE_NAME', 
                'action': 'store',
            },
        ]

        self.add_subcommand()

    def run(self, args):
        #TODO code for defined flags
