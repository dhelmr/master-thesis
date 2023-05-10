from argparse import ArgumentParser

from mlflow import MlflowClient

from tsa.cli.run import SubCommand, make_experiment
from tsa.experiment_checker import ExperimentChecker


class CheckSubCommand(SubCommand):

    def __init__(self):
        super().__init__("check", "check experiment for completeness and integrity")
    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("--experiment", "-e", required=False, help="Sets the mlflow experiment ID", type=str)
        parser.add_argument("-c", "--config", required=True, help="Experiment config yaml file.")

    def exec(self, args, parser):
        mlflow_client = MlflowClient() # TODO global singleton
        experiment = make_experiment(args.config, mlflow_client)
        checker = ExperimentChecker(experiment)
        checker.check_all(args.experiment)