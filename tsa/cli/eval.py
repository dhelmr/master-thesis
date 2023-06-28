import argparse
import os

import pandas as pd
import plotly.express as px
from pandas import DataFrame

from tsa.cli.check import load_exp_from_parser
from tsa.cli.run import SubCommand


class EvalSubCommand(SubCommand):
    def __init__(self):
        super().__init__("eval", "evaluate multiple experiments for their robustness")

    def make_subparser(self, parser: argparse.ArgumentParser):
        parser.add_argument("-c", "--config", required=False, nargs="+",
                            help="Experiment config yaml file. If not set, the config is loaded from the first mlflow run.")
        parser.add_argument("--allow-unfinished", required=False, default=False, action="store_true")

    def exec(self, args, parser):
        pd.options.plotting.backend = "plotly"
        converter = ExperimentNameConversion()
        checkers = []
        for cfg_path in args.config:
            exp_name = converter.infer_exp_name(cfg_path)
            checker = load_exp_from_parser(cfg_path, exp_name)
            checkers.append(checker)
        dfs = []
        for checker in checkers:
            print(checker.experiment.mlflow_name)
            runs = checker.get_runs_df(no_finished_check=args.allow_unfinished)
            runs["params.num_attacks"] = pd.to_numeric(runs["params.num_attacks"])
            aggregated: DataFrame = runs.filter(items=["params.num_attacks",
                                              "metrics.ids.f1_cfa",
                                              "metrics.ids.precision_with_cfa",
                                              "metrics.ids.detection_rate",
                                              "metrics.ids.consecutive_false_positives_normal",
                                              "metrics.ids.consecutive_false_positives_exploits",
                                              "experiment_name",
                                              "metrics.ids.recall"]
                                       ).groupby("params.num_attacks").mean(numeric_only=False).reset_index()
            aggregated.sort_values(by=["params.num_attacks"], inplace=True)
            aggregated["experiment_name"] = converter.get_rel_exp_name(checker.experiment.mlflow_name)
            dfs.append(aggregated)
            # print(group_by)
            # print("===========================\n")
        merged = pd.concat(dfs)
        fig = px.line(merged,
                      x="params.num_attacks",
                      y="metrics.ids.f1_cfa",
                      color="experiment_name",
                      line_dash="experiment_name",
                      line_dash_sequence=["dot"],
                      markers=True)
        fig.show()


class ExperimentNameConversion:
    def __init__(self):
        if "EXPERIMENT_PREFIX" not in os.environ:
            raise KeyError("$EXPERIMENT_PREFIX is not set")
        self.mlflow_exp_prefix = os.environ["EXPERIMENT_PREFIX"]
        if "EXPERIMENT_BASE_PATH" not in os.environ:
            raise KeyError("$EXPERIMENT_BASE_PATH is not set")
        self.exp_base_path = os.path.abspath(os.environ["EXPERIMENT_BASE_PATH"])

    def infer_exp_name(self, config_path: str):
        rel_exp_path = os.path.abspath(config_path)[len(self.exp_base_path) + 1:]
        exp_name = "%s/%s" % (self.mlflow_exp_prefix, rel_exp_path.replace("/", "-"))
        return exp_name

    def get_rel_exp_name(self, mlflow_name: str):
        return mlflow_name[len(self.mlflow_exp_prefix):]
