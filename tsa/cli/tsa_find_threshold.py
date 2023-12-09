from argparse import ArgumentParser
from statistics import mean, median

import pandas
import pandas as pd
from tqdm import tqdm

from tsa.cli.run import SubCommand
from tsa.cli.tsa_cv import load_data
from tsa.perf_pred.cv import CV
from tsa.perf_pred.heuristics import Threshold


class TSAFindThresholdSubCommand(SubCommand):

    def __init__(self):
        super().__init__("tsa-find-threshold", "print stats for training set characteristics")

    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("-i", "--input", required=True,
                            help="input data file (training set statistics -> performance)")
        parser.add_argument("--cv-data", required=True)
        parser.add_argument("--features", "-f", required=False, nargs="+", default=None)
        parser.add_argument("--target", required=True)
        parser.add_argument("--start", default=0, type=float)
        parser.add_argument("--end", default=1, type=float)
        parser.add_argument("--step", default=0.001, type=float)
        parser.add_argument("--scenario-column", default="scenario")
        parser.add_argument("--output", "-o", required=False, help="Output stats csv file")

    def exec(self, args, parser, unknown_args):
        data = load_data(args.input, args.scenario_column, args.features)
        cv_data = load_data(args.cv_data, args.scenario_column, args.features)
        leave_out = len(data.get_scenarios())-1

        stats = []
        for threshold in tqdm(float_range(args.start, args.end, args.step)):
            cv = CV(
                cv_data.with_features([args.target]),
                predictor=Threshold(["--threshold-feature", args.target, "--threshold-lt", "--threshold-value", str(threshold)]),
                cv_leave_out=leave_out
            )
            cv_stats = cv.run("f1_cfa", 0.8, reverse_classes=False)
            stat = self.make_stats(data, threshold, args.target)
            stat["balance_cost"] = self.calc_costs(stat)
            stat = {
                **stat,
                **cv_stats
            }
            stats.append(stat)
        stats = pandas.DataFrame(stats)
        stats = stats.sort_values(by=["balance_cost"])
        stats.to_csv(args.output)



    def make_stats(self, data, threshold, target):
        n_zeros = 0
        min_fractions = []
        class_a_total = 0
        class_b_total = 0
        for sc in data.get_scenarios():
            sc_data = data.get_scenario_data(sc)
            unique_timepoints = pd.unique(sc_data["syscalls"])
            threshold_values = sc_data[target].apply(lambda x: 1 if x > threshold else 0).tolist()
            class_a = sum(threshold_values)
            class_b = len(threshold_values) - sum(threshold_values)
            if class_a == 0:
                n_zeros += 1
            if class_b == 0:
                n_zeros += 1
            min_fraction = get_min_fraction(class_a, class_b)
            min_fractions.append(min_fraction)
            class_a_total += class_a
            class_b_total += class_b
        return {
            "threshold": threshold,
            "n_zeros": n_zeros,
            "mean_min_fractions": mean(min_fractions),
            "median_min_fractions": median(min_fractions),
            "total_min_fraction": get_min_fraction(class_a_total, class_b_total)
        }

    def calc_costs(self, stats):
        if stats["n_zeros"] > 5:
            return 1
        return 0.5-stats["total_min_fraction"]


def float_range(start, end, step):
    current = start
    while current < end:
        yield current
        current += step

def get_min_fraction(a,b):
    return min([a / (a + b), b / (a + b)])