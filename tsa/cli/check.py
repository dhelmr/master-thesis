from argparse import ArgumentParser

from tsa.cli.run import SubCommand


class CheckSubCommand(SubCommand):

    def __init__(self):
        super().__init__("check", "check experiment for completeness and integrity")
    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("--experiment", "-e", required=False, help="Sets the mlflow experiment ID", type=str)
        parser.add_argument("-c", "--config", required=True, help="Experiment config yaml file.")

    def exec(self, args, parser):
        print("TODO CHECK")