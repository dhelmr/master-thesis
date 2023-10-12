import os
import sys
import tempfile
from argparse import ArgumentParser

import mlflow
import pandas
import pandas as pd
from mlflow import MlflowClient

from tsa.cli.run import SubCommand, make_experiment, make_experiment_from_path
from tsa.experiment_checker import ExperimentChecker
from tsa.perf_pred.cv import CV, PerformanceData
from tsa.perf_pred.decision_tree import DecisionTree
from tsa.perf_pred.heuristics import BaselineRandom, BaselineAlways0, BaselineAlways1, BaselineMajorityClass, Heuristic1, Heuristic2

PREDICTORS = {
    cls.__name__: cls for cls in
    [Heuristic1, Heuristic2, BaselineRandom, BaselineAlways1, BaselineAlways0, BaselineMajorityClass, DecisionTree]
}


class TSACrossValidateSubCommand(SubCommand):

    def __init__(self):
        super().__init__("tsa-cv", "cross validate performance predictor")
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("-p", "--predictor", help="Name of the Predictor(s)", nargs="+", choices=PREDICTORS.keys(),
                            default=list(PREDICTORS.keys()))
        parser.add_argument("--all", action="store_true", default=False)
        parser.add_argument("--leave-out", default=2, type=int)
        parser.add_argument("-i", "--input", required=True,
                            help="input data file (training set statistics -> performance)")
        parser.add_argument("--features", "-f", required=True, nargs="+")
        parser.add_argument("--target", default="f1_cfa")
        parser.add_argument("--threshold", default=0.8, type=float)
        parser.add_argument("--scenario-column", default="scenario")
        parser.add_argument("--out", "-o", required=True)

    def exec(self, args, parser):
        data = load_data(args.input, args.features, args.scenario_column)
        all_stats = []
        for predictor_name in args.predictor:
            predictor = PREDICTORS[predictor_name]()
            cv = CV(
                data,
                predictor=predictor,
                cv_leave_out=args.leave_out
            )
            stats = cv.run(args.target, args.threshold).to_dict()
            stats["predictor"] = predictor_name
            stats["threshold"] = args.threshold
            all_stats.append(stats)
        df = pd.DataFrame(all_stats)
        df.to_csv(args.out)
        self._print_results(df)

    def _print_results(self, df: pandas.DataFrame):
        df = df.drop(columns=[c for c in df.columns if c not in ["mcc", "precision", "f1_score", "balanced_accuracy", "predictor"]])
        df.sort_values(by="precision")
        print(df)

def load_data(path: str, feature_cols, scenario_col) -> PerformanceData:
    df = pandas.read_csv(path)
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    df = df.drop(columns=drop_cols)
    for f in feature_cols:
        if f not in df.columns:
            raise ValueError("Feature Column is not available: %s" % f)
    return PerformanceData(
        df=df,
        feature_cols=feature_cols,
        scenario_col=scenario_col
    )
