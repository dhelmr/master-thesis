import os
import pprint
import sys
import tempfile
from argparse import ArgumentParser
from typing import Optional, List

import mlflow
import pandas
import pandas as pd
from mlflow import MlflowClient

from tsa.cli.run import SubCommand, make_experiment, make_experiment_from_path
from tsa.cli.tsa_cv import load_data
from tsa.confusion_matrix import ConfusionMatrix

from tsa.experiment_checker import ExperimentChecker
from tsa.perf_pred.cv import CV, PerformanceData
from tsa.perf_pred.decision_tree import DecisionTree
from tsa.perf_pred.heuristics import BaselineRandom, BaselineAlways0, BaselineAlways1, BaselineMajorityClass, Heuristic1, Heuristic2

PREDICTORS = {
    cls.__name__: cls for cls in
    [Heuristic1, Heuristic2, BaselineRandom, BaselineAlways1, BaselineAlways0, BaselineMajorityClass, DecisionTree]
}

NON_FEATURE_COLS = [
    "syscalls", "run_id", "iteration", "parameter_cfg_id", "num_attacks", "permutation_id", "scenario", "f1_cfa",
    "precision_with_cfa", "recall", "detection_rate"
]

class TSARuleMinerSubCommand(SubCommand):

    def __init__(self):
        super().__init__("tsa-ruleminer", "get rules from performance predictor", expect_unknown_args=True)
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("-p", "--predictor", help="Name of the Predictor", choices=PREDICTORS.keys(),
                            default=list(PREDICTORS.keys()))
        parser.add_argument("-i", "--input", required=True,
                            help="input data file (training set statistics -> performance)")
        parser.add_argument("--features", "-f", required=True, nargs="+", default=None)
        parser.add_argument("--target", default="f1_cfa")
        parser.add_argument("--threshold", default=0.8, type=float)
        parser.add_argument("--scenario-column", default="scenario")
        parser.add_argument("--out", "-o", required=True)

    def exec(self, args, parser, unknown_args):
        data = load_data(args.input, args.scenario_column, args.features)

        split = data.get_split(args.target, [], args.threshold)
        predictor_name = args.predictor
        predictor = PREDICTORS[predictor_name](unknown_args)
        predictor.train(split.train_X, split.train_y)
        preds = predictor.predict(split.train_X) # evalute on same data
        cm = ConfusionMatrix.from_predictions(preds, split.train_y, labels=[0, 1])
        metrics = cm.calc_unweighted_measurements()

        pprint.pprint(metrics)
        print(predictor.extract_rules())