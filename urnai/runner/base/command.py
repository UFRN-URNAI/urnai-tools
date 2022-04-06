import abc


class Command(abc.ABC):

    def __init__(self, top_parser, subparser):
        self.top_parser = top_parser
        self.subparser = subparser

    @abc.abstractmethod
    def run(self, args):
        pass

    def add_subcommand(self):
        subcommand = self.subparser.add_parser(self.command, help=self.help)

        for flag in self.flags:
            subcommand.add_argument(flag['command'], help=flag['help'])

        subcommand.set_defaults(func=self.run)
