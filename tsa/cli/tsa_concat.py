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

class TSAConcatSubCommand(SubCommand):

    def __init__(self):
        super().__init__("tsa-concat", "concat training set characteristics data files")

    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("-i", "--input", required=True,
                            help="input data files (training set statistics -> performance)", nargs="+")
        parser.add_argument("-o", "--output", help="Output csv file")
        parser.add_argument("--common", help="Common column names", nargs="+", default=["f1_cfa", "detection_rate", "precision_with_cfa"])
        parser.add_argument("--skip", nargs="+", default=[])

    def exec(self, args, parser, unknown_args):
        merge_keys = ["scenario", "permutation_i", "iteration", "is_last", "syscalls", "num_attacks"]
        for key in args.common:
            if key not in merge_keys:
                merge_keys.append(key)
        merged_df = None
        for i_file in args.input:
            df = pandas.read_csv(i_file)
            drop_cols = ["parameter_cfg_id", "run_id"]
            for c in args.skip:
                if c in df.columns:
                    drop_cols.append(c)
            df.drop(columns=drop_cols, inplace=True)
            if merged_df is None:
                merged_df = df
                continue
            cols_to_use = df.columns.difference(merged_df.columns).to_list()
            cols_to_use += merge_keys
            merged_df = pandas.merge(merged_df, df[cols_to_use], on=merge_keys, suffixes=("",""))
        merged_df.to_csv(args.output)



