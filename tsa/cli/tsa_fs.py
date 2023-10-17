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


class ForwardSelector:

    def __init__(self, all_data: PerformanceData, selection_metric="mcc", total=5):
        self.all_data = all_data
        self.all_features = all_data.feature_cols()
        self._selection_metric = selection_metric
        self.total = total

    def next(self, last_performances: Optional[List[Tuple[FeatureSelection, CVPerformance]]]) -> Optional[List[Any]]:
        if last_performances is None:
            best_last_round = []
        else:
            best_last_round = None
            best_metric = None
            for features, perf in last_performances:
                curr_metric = perf[self._selection_metric]
                if best_last_round is None or curr_metric > best_metric:
                    best_last_round = features
                    best_metric = curr_metric
        if best_last_round is None:
            raise RuntimeError("Invalid performances")
        if len(best_last_round) == self.total or len(best_last_round) == len(self.all_features):
            return None
        next_features = []
        for f in self.all_features:
            next_features.append(best_last_round + [f])
        return next_features

    def total_rounds(self):
        return self.total

class AllCombinations:
    def __init__(self, all_data: PerformanceData, total=2):
        self.all_data = all_data
        self.all_features = all_data.feature_cols()
        self.n_combinations = len(list(itertools.combinations(self.all_features, total)))
        self._iterator = itertools.combinations(self.all_features, total)

    def next(self, last_performances: Optional[List[Tuple[FeatureSelection, CVPerformance]]]) -> Optional[List[Any]]:
        try:
            next = list(self._iterator.__next__())
            return [next]
        except StopIteration as e:
            return None

    def total_rounds(self):
        return self.n_combinations

FEATURE_SELECTORS = {
    "forward": ForwardSelector,
    "all-2": AllCombinations # TODO find way to set params via cli
}


class TSAFsSubCommand(SubCommand):

    def __init__(self):
        super().__init__("tsa-fs", "perform feature selection for tsa-cv")

    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("-p", "--predictor", help="Name of the Predictor", choices=PREDICTORS.keys())
        parser.add_argument("--all", action="store_true", default=False)
        parser.add_argument("--leave-out", default=2, type=int)
        parser.add_argument("-i", "--input", required=True,
                            help="input data file (training set statistics -> performance)")
        parser.add_argument("--target", default="f1_cfa")
        parser.add_argument("--threshold", default=0.8, type=float)
        parser.add_argument("--features", "-f", required=False, nargs="+", default=None)
        parser.add_argument("--scenario-column", default="scenario")
        parser.add_argument("--out", "-o", required=True)
        parser.add_argument("--mode", "-m", choices=FEATURE_SELECTORS.keys(), default="forward")

    def exec(self, args, parser, unknown_args):
        data = load_data(args.input, args.scenario_column, args.features)
        predictor = PREDICTORS[args.predictor]()
        selector = FEATURE_SELECTORS[args.mode](data)

        all_stats = []
        next_features = selector.next(None)
        i = 0
        while True:
            i += 1
            print("Start round %s/%s" % (i, selector.total_rounds()))
            results = []
            for features in tqdm(next_features):
                cv = CV(
                    data.with_features(features),
                    predictor=predictor,
                    cv_leave_out=args.leave_out
                )
                stats = cv.run(args.target, args.threshold).to_dict()
                results.append((features, stats,))
                stats["predictor"] = args.predictor
                stats["threshold"] = args.threshold
                stats["features"] = str(features)
                all_stats.append(stats)

            print("Best round results:")
            round_results_df = pd.DataFrame([
                stats for _, stats in results
            ])
            print_results(round_results_df, limit=3, cols=["mcc", "f1_score", "precision", "features"])

            next_features = selector.next(results)
            if next_features is None:
                break
        df = pd.DataFrame(all_stats)
        df.to_csv(args.out)
        print_results(df, cols=["mcc", "f1_score", "precision", "features"])
