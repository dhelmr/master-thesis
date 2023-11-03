import os
import sys
import tempfile
from argparse import ArgumentParser

import mlflow
import pandas
import pandas as pd
from mlflow import MlflowClient
from tqdm import tqdm

from tsa.cli.run import SubCommand, make_experiment, make_experiment_from_path
from tsa.cli.tsa_combine import METRIC_KEYS
from tsa.cli.tsa_cv import load_data
from tsa.experiment_checker import ExperimentChecker
from tsa.mlflow.experiment_name_conversion import ExperimentNameConversion

class TSAStatsSubCommand(SubCommand):

    def __init__(self):
        super().__init__("tsa-stats", "print stats for training set characteristics")

    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("-i", "--input", required=True,
                            help="input data file (training set statistics -> performance)")
        parser.add_argument("--features", "-f", required=False, nargs="+", default=None)
        parser.add_argument("--targets", default=["f1_cfa", "precision_with_cfa", "detection_rate"], nargs="+")
        parser.add_argument("--threshold", default=0.8, type=float)
        parser.add_argument("--scenario-column", default="scenario")

    def exec(self, args, parser, unknown_args):
        data = load_data(args.input, args.scenario_column, args.features)
        stats = []
        for sc in data.get_scenarios():
            sc_data = data.get_scenario_data(sc)
            unique_timepoints = pd.unique(sc_data["syscalls"])
            stat = {
                "scenario": sc,
                "unique_timepoints": len(unique_timepoints),
                "rows": len(sc_data)
            }
            for t in args.targets:
                stat[f"{t}-min"] = sc_data[t].min()
                stat[f"{t}-max"] = sc_data[t].max()
                stat[f"{t}-mean"] = sc_data[t].mean()
                stat[f"{t}-range"] = stat[f"{t}-max"] - stat[f"{t}-min"]
                threshold_values = sc_data[t].apply(lambda x: 1 if x>args.threshold else 0).tolist()
                stat[f"{t}>{args.threshold}"] = sum(threshold_values)
                stat[f"{t}<={args.threshold}"] = len(threshold_values) - sum(threshold_values)
            stats.append(stat)
        stats = pandas.DataFrame(stats)
        stats.sort_values(by="unique_timepoints", inplace=True)
        print(stats)
        print("Scenarios:", len(data.get_scenarios()))
        print("\n======================================")
        print("avg values (across scenarios):")
        print(stats.mean())


