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
from tsa.cli.tsa_ruleminer import TSARuleMinerSubCommand
from tsa.experiment_checker import ExperimentChecker
from tsa.perf_pred.cv import CV, CVPerformance, PerformanceData

FeatureSelection = List[str]


class TSAEvalFsSubCommand(SubCommand):

    def __init__(self):
        super().__init__("tsa-eval-fs", "evaluate feature selection results", expect_unknown_args=True)

    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("-f", "--feature-file", required=True,
                            help="input feature-selection file (feature selection CSV)")
        parser.add_argument("--out", "-o", required=True)
        parser.add_argument("--input", help="Performance Data CSV", required=True)
        parser.add_argument("--target", default="f1_cfa")
        parser.add_argument("--threshold", default=0.8, type=float)
        parser.add_argument("--reverse-classes", default=False, action="store_true")
        parser.add_argument("--scenario-column", default="scenario")
        parser.add_argument("-p", "--predictor", help="Name of the Predictor", choices=PREDICTORS.keys(),
                            default=list(PREDICTORS.keys()))
        parser.add_argument("-q", "--query", help="Query for filtering the feature selection set",
                            default="`gain.precision` > 0.01 and `mean.f1_score` > 0.5")
        parser.add_argument("--sort-by", help="Sort feature selection set by ...",
                            default="mean.precision")

    def exec(self, args, parser, unknown_args):
        data, rules_miner = TSARuleMinerSubCommand.init_rulesminer(args, unknown_args)

        fs_results = pandas.read_csv(args.feature_file, index_col=False)
        fs_results["key"] = fs_results.apply(lambda r: self._get_row_key(r["features"].split(";")), axis=1)
        fs_results.set_index("key", inplace=True)

        fs_results["gain.precision"] = fs_results.apply(
            lambda r: self._calc_gain(r, fs_results, variable="mean.precision"), axis=1)
        # fs_results["gain"] = fs_results.apply(lambda row:
        # fs_results.to_csv(args.out)
        fs_results = fs_results.query(args.query)
        fs_results.sort_values(by="mean.precision", ascending=False, inplace=True)
        for i, features in enumerate(fs_results[:5]["features"]):
            feature_set = features.split(";")
            print(feature_set)
            split = data.with_features(feature_set).get_split(args.target, [], args.threshold, args.reverse_classes)
            svg_path = os.path.join(args.out, f"{i}.png")
            rules_miner.extract_rules(split, svg_path)

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
