import os
from argparse import ArgumentParser
from typing import List

import pandas
import pandas as pd
from tqdm import tqdm

from tsa.cli.run import SubCommand
from tsa.cli.tsa_cv import PREDICTORS
from tsa.cli.tsa_ruleminer import TSARuleMinerSubCommand

FeatureSelection = List[str]


class TSAEvalFsSubCommand(SubCommand):

    def __init__(self):
        super().__init__("tsa-eval-fs", "evaluate feature selection results", expect_unknown_args=True)

    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("-f", "--feature-files", required=True, nargs="+",
                            help="input feature-selection file(s) (feature selection CSV)")
        parser.add_argument("--out", "-o", required=True)
        parser.add_argument("--input", help="Performance Data CSV", required=True)
        parser.add_argument("--target", default="f1_cfa")
        parser.add_argument("--threshold", default=0.8, type=float)
        parser.add_argument("--reverse-classes", default=False, action="store_true")
        parser.add_argument("--scenario-column", default="scenario")
        parser.add_argument("-p", "--predictor", help="Name of the Predictor", choices=PREDICTORS.keys(),
                            default="DecisionTree")
        parser.add_argument("-q", "--query", help="Query for filtering the feature selection set",
                            default="`gain.precision` > 0.01 and `mean.f1_score` > 0.5")
        parser.add_argument("--sort-by", help="Sort feature selection set by ...",
                            default="mean.precision")
        parser.add_argument("--svgs", required=False, default=False, action="store_true")
        parser.add_argument("--add-max-depth-column", required=False, default=False, action="store_true",
                            help="If set, the max-depth for the decision tree in feature selection method is parsed from the file name.")

    def exec(self, args, parser, unknown_args):
        data, rules_miner = TSARuleMinerSubCommand.init_rulesminer(args, unknown_args)

        li = []
        for f in tqdm(args.feature_files):
            fs_results = pandas.read_csv(f, index_col=False)
            file_basename = os.path.basename(f)
            fs_results["file"] = file_basename
            fs_results["key"] = fs_results.apply(lambda r: self._get_row_key(r["features"].split(";")), axis=1)
            fs_results.set_index("key", inplace=True)
            fs_results["gain.precision"] = fs_results.apply(
                lambda r: self._calc_gain(r, fs_results, variable="mean.precision"), axis=1)
            if args.add_max_depth_column:
                fs_results["max_depth"] = self._parse_max_depth_decision_tree(file_basename)
            li.append(fs_results)

        fs_results = pd.concat(li, axis=0, ignore_index=True)

        fs_results = fs_results.query(args.query)
        fs_results.sort_values(by="mean.precision", ascending=False, inplace=True)
        out_path = os.path.join(args.out, "results.csv")
        fs_results.to_csv(out_path)

        if args.svgs:
            for i, features in enumerate(fs_results[:5]["features"]):
                print("Precision Gain:", fs_results.iloc[i]["gain.precision"])
                feature_set = features.split(";")
                print(feature_set)
                split = data.with_features(feature_set).get_split(args.target, [], args.threshold, args.reverse_classes)
                svg_path = os.path.join(args.out, f"{i}.png")
                rules_miner.extract_rules(split, svg_path)

    def _get_row_key(self, features):
        key = list(sorted(features))
        return " ".join(key)

    def _calc_gain(self, row, fs_results, variable="mean.precision"):
        features = row["features"].split(";")
        last_features = features[:-1]
        if len(last_features) == 0:
            gain = row[variable]
        else:
            last_row = fs_results.loc[self._get_row_key(last_features)]
            gain = row[variable] - last_row[variable]
        return gain

    def _parse_max_depth_decision_tree(self, file_name: str):
        # expected file format is: "*-depth=%{depth}rounds=*.csv"
        return int(substring(file_name, "-depth=", "rounds="))

def substring(s, before, after):
    return s[s.index(before)+len(before):s.index(after)]