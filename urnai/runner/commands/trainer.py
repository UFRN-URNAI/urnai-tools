import os

from urnai.runner.base.command import Command
from urnai.utils.reporter import Reporter as rp 

class TrainerCommand(Command):

    def __init__(self, top_parser, subparser):
        super().__init__(top_parser, subparser)
        self.command = 'train'
        self.help = 'Subcommand used to start trainings.'
        self.flags = [
                {'command': '--train-file', 'help': 'JSON or CSV solve file, with all the parameters to start the training.', 'type' : str, 'metavar' : 'TRAIN_FILE_PATH', 'action' : 'store'},
    #TODO            {'command': '--build-training-file', 'help': 'Helper to build a solve json-file.', 'action' : 'store_true'},
                {'command': '--convert', 'help': 'Training file to convert. Must be used with --output-format option.', 'action' : 'store'},
                {'command': '--output-format', 'help': 'Converted file format . Must be used with --convert option. Accepted values are \'CSV\', \'YAML\' and \'JSON\'.', 'action' : 'store'},
                {'command': '--play', 'help': 'Test agent, without training it, it will ignore train entry on json file.', 'action' : 'store_true'},
                {'command': '--threaded', 'help': 'If there is more than one training in the file, it will be executed in a separated thread.', 'action' : 'store_true'},
                ]

        self.add_subcommand()

    def run(self, args):
        if args.train_file is not None:
            from urnai.trainers.filetrainer import FileTrainer
            trainer = FileTrainer(args.train_file)

            #this is needed in case the environment is starcraft
            for arg in args.__dict__.keys():
                arg = "--{}".format(arg.replace("_", "-"))
                if arg in sys.argv: sys.argv.remove(arg)

            trainer.start_training(play_only=args.play, threaded_training=args.threaded)
        #TODO
        #elif args.build_training_file:
        elif args.convert is not None:
            if args.output_format is not None:
                from urnai.trainers.filetrainer import FileTrainer
                trainer = FileTrainer(args.convert)
                trainer.check_trainings()
                output_path = os.path.abspath(os.path.dirname(args.convert))+ os.path.sep + os.path.splitext(os.path.basename(args.convert))[0] + ".file_format"
                output_text = "{} was converted to {}.".format(
                        os.path.basename(args.convert),
                        os.path.basename(output_path)
                        )

                if args.output_format == 'CSV':
                    trainer.save_trainings_as_csv(output_path.replace('.file_format', '.csv'))
                    rp.report(output_text.replace('.file_format', '.csv'))
                elif args.output_format == 'JSON':
                    trainer.save_trainings_as_json(output_path.replace('.file_format', '.json'))
                    rp.report(output_text.replace('.file_format', '.json'))
                elif args.output_format == 'YAML':
                    trainer.save_trainings_as_yaml(output_path.replace('.file_format', '.yaml'))
                    rp.report(output_text.replace('.file_format', '.yaml'))
                else:
                    raise Exception("--out-format must be 'CSV', 'JSON' or 'YAML'.")
            else:
                raise Exception("You must specify --output-format.")
        else:
            raise Exception("You must specify --train-file or --convert (with --output-format set).")
