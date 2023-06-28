import argparse

import mlflow
from mlflow import MlflowClient

from tsa.cli.check import load_exp_from_parser
from tsa.cli.run import SubCommand


class EvalSubCommand(SubCommand):
    def __init__(self):
        super().__init__("eval", "evaluate multiple experiments for their robustness")

    def make_subparser(self, parser: argparse.ArgumentParser):
        parser.add_argument("--experiment", "-e", required=False, help="Sets the mlflow experiment ID", type=str)
        parser.add_argument("-c", "--config", required=False,
                            help="Experiment config yaml file. If not set, the config is loaded from the first mlflow run.")
        parser.add_argument("--allow-unfinished", required=False, default=False, action="store_true")

    def exec(self, args, parser):
        checker = load_exp_from_parser(args.config, args.experiment)
        runs = checker.get_runs_df(no_finished_check=args.allow_unfinished)
        group_by = runs.filter(items=["params.num_attacks",
                                      "metrics.ids.f1_cfa",
                                      "metrics.ids.precision_with_cfa",
                                      "metrics.ids.detection_rate",
                                      "metrics.ids.consecutive_false_positives_normal",
                                      "metrics.ids.consecutive_false_positives_exploits",
                                      "metrics.ids.recall"]
                               ).groupby("params.num_attacks").mean()
        print(group_by)
