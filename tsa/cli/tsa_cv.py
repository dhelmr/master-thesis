from argparse import ArgumentParser
from argparse import ArgumentParser
from typing import Optional, List

import pandas
import pandas as pd

from tsa.cli.run import SubCommand
from tsa.perf_pred.cv import CV, PerformanceData
from tsa.perf_pred.decision_tree import DecisionTree
from tsa.perf_pred.heuristics import BaselineRandom, BaselineAlways0, BaselineAlways1, BaselineMajorityClass, \
    Heuristic1, Heuristic2, Threshold
from tsa.perf_pred.logistic_regression import LogisticRegression
from tsa.perf_pred.random_forrest import RandomForrest

PREDICTORS = {
    cls.__name__: cls for cls in
    [Heuristic1, Heuristic2, BaselineRandom, BaselineAlways1, BaselineAlways0, BaselineMajorityClass, DecisionTree,
     LogisticRegression, RandomForrest, Threshold]
}

NON_FEATURE_COLS = [
    "syscalls", "run_id", "iteration", "parameter_cfg_id", "num_attacks", "permutation_i", "scenario", "f1_cfa",
    "precision_with_cfa", "recall", "detection_rate", "is_last"
]

class TSACrossValidateSubCommand(SubCommand):

    def __init__(self):
        super().__init__("tsa-cv", "cross validate performance predictor", expect_unknown_args=True)
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("-p", "--predictor", help="Name of the Predictor(s)", nargs="+", choices=PREDICTORS.keys(),
                            default=list(PREDICTORS.keys()))
        parser.add_argument("--all", action="store_true", default=False)
        parser.add_argument("--leave-out", default=2, type=int)
        parser.add_argument("-i", "--input", required=True,
                            help="input data file (training set statistics -> performance)")
        parser.add_argument("--features", "-f", required=False, nargs="+", default=None)
        parser.add_argument("--target", default="f1_cfa")
        parser.add_argument("--threshold", default=0.8, type=float)
        parser.add_argument("--reverse-classes", default=False, action="store_true")
        parser.add_argument("--scenario-column", default="scenario")
        parser.add_argument("--skip-features", "-s", required=False, nargs="+", default=[])
        parser.add_argument("--out", "-o", required=True)

    def exec(self, args, parser, unknown_args):
        data = load_data(args.input, args.scenario_column, args.features, skip_features=args.skip_features)
        all_stats = []
        for predictor_name in args.predictor:
            predictor = PREDICTORS[predictor_name](unknown_args)
            cv = CV(
                data,
                predictor=predictor,
                cv_leave_out=args.leave_out
            )
            stats = cv.run(args.target, args.threshold, args.reverse_classes)
            stats["predictor"] = predictor_name
            stats["threshold"] = args.threshold
            all_stats.append(stats)
        df = pd.DataFrame(all_stats)
        df.to_csv(args.out)
        print_results(df)

def print_results(df: pandas.DataFrame, limit=None, cols=None):
    if cols is None:
        cols = ["mean.mcc", "var.mcc", "mean.precision", "var.precision", "mean.f1_score", "var.f1_score", "mean.balanced_accuracy", "predictor"]
    df = df.drop(columns=[c for c in df.columns if c not in cols])
    df.sort_values(by="mean.precision", inplace=True, ascending=False)
    if limit is not None:
        df = df.iloc[:limit]
    print(df)

def load_data(path: str, scenario_col, feature_cols: Optional[List[str]], skip_features=[]) -> PerformanceData:
    df = pandas.read_csv(path)
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    df = df.drop(columns=drop_cols)
    if feature_cols is None:
        # all columns without a "." and not in above list are considered feature cols
        feature_cols = [c for c in df.columns if str(c) not in NON_FEATURE_COLS and str(c) not in skip_features and "." not in str(c)]
        #print("Selected features:", feature_cols)
    for f in feature_cols:
        if f not in df.columns:
            raise ValueError("Feature Column is not available: %s" % f)
    return PerformanceData(
        df=df,
        feature_cols=feature_cols,
        scenario_col=scenario_col
    )
