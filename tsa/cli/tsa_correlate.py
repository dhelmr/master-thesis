import math
import os
import pprint
import sys
import tempfile
from argparse import ArgumentParser

import mlflow
import pandas
from mlflow import MlflowClient

from tsa.cli.run import SubCommand, make_experiment, make_experiment_from_path
from tsa.cli.tsa_cv import load_data
from tsa.experiment_checker import ExperimentChecker


class TSACorrelateSubCommand(SubCommand):

    def __init__(self):
        super().__init__("tsa-correlate", "show correlations of training set statistics")
    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("-i", "--input", required=True,
                            help="input data file (training set statistics -> performance)")
        parser.add_argument("--features", "-f", required=False, nargs="+", default=None)
        parser.add_argument("--target", default="f1_cfa")
       # parser.add_argument("--threshold", default=0.8, type=float)
        parser.add_argument("--scenario-column", default="scenario")
        parser.add_argument("--only-above", default=0.2)
        parser.add_argument("-o", "--output", default=None, type=str)

    def exec(self, args, parser, unknown_args):
        data = load_data(args.input, args.scenario_column, args.features)
        corr = data.df.corrwith(data.df[args.target])
        corr = corr.apply(lambda x: abs(x))
        corr.apply(lambda x: math.nan if x < args.only_above else x)
        corr.dropna(inplace=True)
        corr.sort_values(inplace=True)
        as_list = list(zip(corr.index, corr))
        if args.output is not None:
            df = pandas.DataFrame(as_list, columns=["feature", "corr_coeff"])
            df.to_csv(args.output, index=False)
        pprint.pprint(as_list)
