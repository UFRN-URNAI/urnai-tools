from urnai.runner.base.command import Command
from urnai.utils.constants import Trainings
from urnai.utils.module_specialist import get_classes_recursively, get_class_parameters
from urnai.utils.reporter import Reporter as rp
import yaml


class GetCommand(Command):
    def __init__(self, top_parser, subparser):
        super().__init__(top_parser, subparser)
        self.command = 'get'
        self.help = 'Subcommand used to get info about Urnai parts.'
        self.flags = [{'command': '--environments',
                       'help': 'List available environments.',
                       'action': 'store_true',
                       },
                      {'command': '--environment',
                       'help': 'Show parameter info about the passed ENVIRONMENT.',
                       'type': str,
                       'metavar': 'ENVIRONMENT',
                       'action': 'store',
                       },
                      {'command': '--agents',
                       'help': 'List available agents.',
                       'action': 'store_true',
                       },
                      {'command': '--agent',
                       'help': 'Show parameter info about the passed AGENT.',
                       'type': str,
                       'metavar': 'AGENT',
                       'action': 'store',
                       },
                      {'command': '--action-wrappers',
                       'help': 'List available action-wrappers.',
                       'action': 'store_true',
                       },
                      {'command': '--action-wrapper',
                       'help': 'Show parameter info about the passed ACTION_WRAPPER.',
                       'type': str,
                       'metavar': 'ACTION_WRAPPER',
                       'action': 'store',
                       },
                      {'command': '--state-builders',
                       'help': 'List available state-builders.',
                       'action': 'store_true',
                       },
                      {'command': '--state-builder',
                       'help': 'Show parameter info about the passed STATE_BUILDER.',
                       'type': str,
                       'metavar': 'STATE_BUILDER',
                       'action': 'store',
                       },
                      {'command': '--rewards',
                       'help': 'List available state-builders.',
                       'action': 'store_true',
                       },
                      {'command': '--reward',
                       'help': 'Show parameter info about the passed REWARD.',
                       'type': str,
                       'metavar': 'REWARD',
                       'action': 'store',
                       },
                      {'command': '--models',
                       'help': 'List available models.',
                       'action': 'store_true',
                       },
                      {'command': '--model',
                       'help': 'Show parameter info about the passed MODEL.',
                       'type': str,
                       'metavar': 'MODEL',
                       'action': 'store',
                       },
                      {'command': '--trainer',
                       'help': 'Show parameter info about Trainer.',
                       'action': 'store_true',
                       },
                      {'command': '--default-training-file',
                       'help': 'Generate a base training file in YAML format, ' +
                       'redirect the output to a file to save or use --save FILE_NAME',
                       'action': 'store_true',
                       },
                      {'command': '--dummy-training-file',
                       'help': 'Generate a dummy training file in YAML format, ' +
                       'redirect the output to a file to save or use --save FILE_NAME',
                       'action': 'store_true',
                       },
                      {'command': '--save',
                       'help': 'File name to save output to.',
                       'type': str,
                       'metavar': 'FILE_NAME',
                       'action': 'store',
                       },
                      ]

        self.add_subcommand()


    def run(self, args):

        #environments
        if args.environments:
            output = self.beautiful_class_list('Environments', 
                    get_classes_recursively('urnai.envs', ignore=['Env']))
            rp.report(output, clean=True)
        elif args.environment is not None:
            output = self.yml_class_output(get_class_parameters('urnai.envs', args.environment), 'env')
            rp.report(output, clean=True)
        #agents
        elif args.agents:
            clss = get_classes_recursively('urnai.agents', ignore=['Agent'])

            filtered = [cls for cls in clss if 'Agent' in cls]

            output = self.beautiful_class_list('Agents', filtered)
            rp.report(output, clean=True)
        elif args.agent is not None:
            output = self.yml_class_output(get_class_parameters('urnai.agents', args.agent), 'agent')
            rp.report(output, clean=True)
        #actionwrappers
        elif args.action_wrappers:
            output = self.beautiful_class_list('ActionWrappers', 
                    get_classes_recursively('urnai.agents.actions', ignore=['ActionWrapper']))
            rp.report(output)
        elif args.action_wrapper is not None:
            output = self.yml_class_output(get_class_parameters('urnai.agents.actions', args.action_wrapper), 'action_wrapper')
            rp.report(output, clean=True)
        #statebuilders
        elif args.state_builders:
            output = self.beautiful_class_list('StateBuilders', 
                    get_classes_recursively('urnai.agents.states', ignore=['StateBuilder']))
            rp.report(output)
        elif args.state_builder is not None:
            output = self.yml_class_output(get_class_parameters('urnai.agents.states', args.state_builder), 'state_builder')
            rp.report(output, clean=True)
        #statebuilders
        elif args.rewards:
            output = self.beautiful_class_list('RewardBuilders', 
                    get_classes_recursively('urnai.agents.rewards', ignore=['RewardBuilder']))
            rp.report(output)
        elif args.reward is not None:
            output = self.yml_class_output(get_class_parameters('urnai.agents.rewards', args.reward), 'reward')
            rp.report(output, clean=True)
        #models
        elif args.models:
            output = self.beautiful_class_list('LearningModels', 
                    get_classes_recursively('urnai.models', ignore=['LearningModel']))
            rp.report(output)
        elif args.model is not None:
            output = self.yml_class_output(get_class_parameters('urnai.models', args.model), 'model')
            rp.report(output, clean=True)
        #trainer
        elif args.trainer:
            output = self.yml_class_output(get_class_parameters('urnai.trainers', 'Trainer'), 'trainer')
            rp.report(output, clean=True)
        #premade training files
        elif args.default_training_file:
            beautiful = yaml.dump(Trainings.DEFAULT_TRAINING, default_flow_style=False)
            if args.save is not None:
                with open(args.save, 'w+') as out_file:
                    out_file.write(beautiful)
            else:
                rp.report(beautiful, clean=True)
        elif args.dummy_training_file:
            beautiful = yaml.dump(Trainings.DUMMY_TRAINING, default_flow_style=False)
            if args.save is not None:
                with open(args.save, 'w+') as out_file:
                    out_file.write(beautiful)
            else:
                rp.report(beautiful, clean=True)
        else:
            print("urnai get: Use -h or --help for help.")

    def beautiful_class_list(self, title, lst):
        string = 'Available ' + title + ':\n'
        for cls in lst:
            string += cls + '\n'

        return string

    def yml_class_output(self, cls_dict, master_key):
        dct = {}
        dct[master_key] = {
                'class': cls_dict['name'],
                'params': {},
                }

        for param in cls_dict['params_without_defaults']:
            dct[master_key]['params'][param] = '__value_needed__'

        for param in cls_dict['params_with_defaults']:
            dct[master_key]['params'][param['param']] = param['default_value']

        string = yaml.dump(dct, default_flow_style=False)
        return string

