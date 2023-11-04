import math
import os
import tempfile
from argparse import ArgumentParser

import mlflow
import pandas
import pandas as pd
from mlflow import MlflowClient
from pandas import DataFrame, Series
from tqdm import tqdm

from tsa.cli.run import SubCommand, make_experiment, make_experiment_from_path
from tsa.cli.tsa_cv import load_data
from tsa.experiment_checker import ExperimentChecker
from tsa.perf_pred.cv import PerformanceData


class TimeSeries:
    def __init__(self):
        self.unknown_slope = math.nan

    def augment(self, data: PerformanceData) -> pandas.DataFrame:
        dfs = []
        for sc in tqdm(data.get_scenarios()):
            df = data.get_scenario_data(sc).copy()
            df.sort_values(by=data.syscalls_col, inplace=True)
            df.drop_duplicates(ignore_index=True, inplace=True, subset=[data.syscalls_col, data.scenario_col])
            for f in data.feature_cols():
                slope = self._calc_slope(df[data.syscalls_col], df[f])
                df[f"roc-{f}"] = slope
                df[f"roc2-{f}"] = self._calc_slope(df[data.syscalls_col],slope)
            df = df.dropna(how="any")
            dfs.append(df)
        augmented_df = pandas.concat(dfs)
        return augmented_df

    def _calc_slope(self, x: Series, y: Series):
        return y.diff() / x.diff()

class FeatureCombine:
    def __init__(self):
        pass

    def augment(self, data: PerformanceData) -> pandas.DataFrame:
        for f1 in tqdm(data.feature_cols()):
            for f2 in data.feature_cols():
                data.df[f"{f1}/{f2}"] = data.df[f1] / data.df[f2]
                data.df[f"{f1}*{f2}"] = data.df[f1] * data.df[f2]
            data.df[f"loge-{f1}"] = data.df[f1].apply(lambda x: math.log(abs(x)) if x != 0 else -100)
            data.df[f"log2-{f1}"] = data.df[f1].apply(lambda x: math.log(abs(x), 2) if x != 0 else -100)
            data.df[f"log10-{f1}"] = data.df[f1].apply(lambda x: math.log(abs(x), 10) if x != 0 else -100)
            # math.log(0.0000000000000000000000000000000000000000001) ~= -100
            data.df[f"sqrt-{f1}"] = data.df[f1].apply(lambda x: math.sqrt(abs(x)))
            data.df[f"pow2-{f1}"] = data.df[f1].apply(lambda x: math.pow(x, 2))
            data.df[f"abs-{f1}"] = data.df[f1].apply(lambda x: abs(x))
        cols_before = list(data.df.columns)
        data.df = data.df.dropna(axis=1, how='any')
        dropped_cols = [c for c in cols_before if str(c) not in set(data.df.columns)]
        print("Dropped the following columns because of nan values:", dropped_cols)
        return data.df


AUGMENTORS = {
    cls.__name__: cls for cls in [TimeSeries, FeatureCombine]
}


class TSAAugmentSubCommand(SubCommand):

    def __init__(self):
        super().__init__("tsa-augment", "analyse training set experiments")

    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("-o", "--output", required=True, help="Output file")
        parser.add_argument("-i", "--input", required=True,
                            help="input data file (training set statistics -> performance)")
        parser.add_argument("--scenario-column", default="scenario")
        parser.add_argument("--features", "-f", required=False, default=None, nargs="+")
        parser.add_argument("--augmentor", "-a", required=True, choices=AUGMENTORS.keys())

    def exec(self, args, parser, unknown_args):
        data = load_data(args.input, args.scenario_column, args.features)
        augmentor = AUGMENTORS[args.augmentor]()
        augmented = augmentor.augment(data)
        augmented.to_csv(args.output)
