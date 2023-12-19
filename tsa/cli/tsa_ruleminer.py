import pprint
import pprint
import typing
from argparse import ArgumentParser

from tsa.cli.run import SubCommand
from tsa.cli.tsa_cv import load_data
from tsa.confusion_matrix import ConfusionMatrix
from tsa.perf_pred.cv import PerformanceData, TrainTestSplit, PerformancePredictor
from tsa.perf_pred.decision_tree import DecisionTree
from tsa.perf_pred.heuristics import BaselineRandom, BaselineAlways0, BaselineAlways1, BaselineMajorityClass, \
    Heuristic1, Heuristic2
from tsa.perf_pred.logistic_regression import LogisticRegression

PREDICTORS = {
    cls.__name__: cls for cls in
    [Heuristic1, Heuristic2, BaselineRandom, BaselineAlways1, BaselineAlways0, BaselineMajorityClass, DecisionTree,
     LogisticRegression]
}

NON_FEATURE_COLS = [
    "syscalls", "run_id", "iteration", "parameter_cfg_id", "num_attacks", "permutation_id", "scenario", "f1_cfa",
    "precision_with_cfa", "recall", "detection_rate"
]


class RulesMiner:
    def __init__(self, predictor: PerformancePredictor, class_names):
        self._predictor = predictor
        self._class_names = class_names

    def extract_rules(self, split: TrainTestSplit, path: str):
        self._predictor.reset()
        self._predictor.train(split.train_X, split.train_y)
        preds = self._predictor.predict(split.train_X)  # evalute on same data
        cm = ConfusionMatrix.from_predictions(preds, split.train_y, labels=[0, 1])
        metrics = cm.calc_unweighted_measurements()

        pprint.pprint(metrics)
        print(self._predictor.extract_rules(path, self._class_names))

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
        parser.add_argument("--reverse-classes", default=False, action="store_true")
        parser.add_argument("--scenario-column", default="scenario")
        parser.add_argument("--out", "-o", required=True)

    def exec(self, args, parser, unknown_args):
        data, rule_miner = TSARuleMinerSubCommand.init_rulesminer(args, unknown_args)
        data = data.with_features(feature_cols=args.features)
        split = data.get_split(args.target, [], args.threshold, args.reverse_classes)
        rule_miner.extract_rules(split, args.out)

    @staticmethod
    def init_rulesminer(args, unknown_args) -> typing.Tuple[PerformanceData, RulesMiner]:
        data = load_data(args.input, args.scenario_column, feature_cols=None)
        class_names = [f"{args.target}<={args.threshold}", f"{args.target}>{args.threshold}"]
        if args.reverse_classes:
            class_names = [f"{args.target}>{args.threshold}", f"{args.target}<={args.threshold}"]
        predictor_name = args.predictor
        predictor = PREDICTORS[predictor_name](unknown_args)
        rule_miner = RulesMiner(predictor, class_names)
        return data, rule_miner

