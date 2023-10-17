import os
import sys
import tempfile
from argparse import ArgumentParser

import mlflow
import pandas
from mlflow import MlflowClient

from tsa.cli.run import SubCommand, make_experiment, make_experiment_from_path
from tsa.experiment_checker import ExperimentChecker


class TSACorrelateSubCommand(SubCommand):

    def __init__(self):
        super().__init__("tsa-correlate", "combine downloaded training set statistics and performance measures")
    def make_subparser(self, parser: ArgumentParser):

        pass
    def exec(self, args, parser, unknown_args):
        pass

