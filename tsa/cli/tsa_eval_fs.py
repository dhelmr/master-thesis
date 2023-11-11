import itertools
import os
import tempfile
from argparse import ArgumentParser
from typing import List, Dict, Optional, Tuple, Any

import mlflow
import pandas
import pandas as pd
from mlflow import MlflowClient
from tqdm import tqdm

from tsa.cli.run import SubCommand, make_experiment, make_experiment_from_path
from tsa.cli.tsa_cv import PREDICTORS, load_data, print_results
from tsa.experiment_checker import ExperimentChecker
from tsa.perf_pred.cv import CV, CVPerformance, PerformanceData

FeatureSelection = List[str]


class TSAEvalFsSubCommand(SubCommand):

    def __init__(self):
        super().__init__("tsa-eval-fs", "evaluate feature selection results", expect_unknown_args=True)

    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("-i", "--input", required=True,
                            help="input data file (feature selection CSV)")
        parser.add_argument("--out", "-o", required=True)
    def exec(self, args, parser, unknown_args):
        fs_results = pandas.read_csv(args.input, index_col=False)
        fs_results["key"] = fs_results.apply(lambda r: self._get_row_key(r["features"].split(";")), axis=1)
        fs_results.set_index("key", inplace=True)

        fs_results["gain.precision"] = fs_results.apply(lambda r: self._calc_gain(r, fs_results, variable="mean.precision"), axis=1)
        #fs_results["gain"] = fs_results.apply(lambda row:
        fs_results.to_csv(args.out)

    def _get_row_key(self, features):
        key = list(sorted(features))
        return " ".join(key)

    def _calc_gain(self, row, fs_results, variable="mean.precision"):
        features = row["features"].split(";")
        last_features = features[:-1]
        if len(last_features) == 0:
            gain = row[variable]
        else:
            last_row = fs_results.loc[self._get_row_key(last_features)]
            gain = row[variable] - last_row[variable]
        return gain