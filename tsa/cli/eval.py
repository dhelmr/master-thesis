import argparse

import pandas as pd
import plotly.express as px
from pandas import DataFrame

from tsa.cli.check import load_exp_from_parser
from tsa.cli.run import SubCommand
from tsa.mlflow.experiment_name_conversion import ExperimentNameConversion, MlflowResultsCache

NUM_ATTACK_WEIGHTS = {  # TODO!
    0: 0,
    1: 1,
    2: 0.5,
    3: 0.25,
    5: 0.125,
    10: 0.125
}


class EvalSubCommand(SubCommand):
    def __init__(self):
        super().__init__("eval", "evaluate multiple experiments for their robustness")

    def make_subparser(self, parser: argparse.ArgumentParser):
        parser.add_argument("-c", "--config", required=False, nargs="+",
                            help="Experiment config yaml file. If not set, the config is loaded from the first mlflow run.")
        parser.add_argument("--allow-unfinished", required=False, default=False, action="store_true")
        parser.add_argument("--cache", default=None, help="If set, use a local directory to cache mlflow results.")

    def exec(self, args, parser):
        pd.options.plotting.backend = "plotly"
        converter = ExperimentNameConversion()
        if args.cache is not None:
            cache = MlflowResultsCache(args.cache)
        else:
            cache = None
        checkers = []
        for cfg_path in args.config:
            exp_name = converter.infer_exp_name(cfg_path)
            checker = load_exp_from_parser(cfg_path, exp_name)
            checkers.append(checker)
        robustness_values = {}
        dfs = []
        for checker in checkers:
            name = converter.get_rel_exp_name(checker.experiment.mlflow_name)
            print(checker.experiment.mlflow_name)
            runs = self._get_results(checker, name, cache, args.allow_unfinished)
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
            robustness_values[name] = self._calc_robustness_score(aggregated)
            aggregated.sort_values(by=["params.num_attacks"], inplace=True)
            aggregated["experiment_name"] = name
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
        print(robustness_values)

    def _get_results(self, checker, exp_name, cache, allow_unfinished):
        if cache is  None:
            runs, _ = checker.get_runs_df(no_finished_check=allow_unfinished)
            return runs

        from_cache = cache.get_cached_result(exp_name)
        if from_cache is None:
            runs, is_finished = checker.get_runs_df(no_finished_check=allow_unfinished)
            if is_finished:
                cache.cache(exp_name, runs)
        else:
            runs = from_cache.df
            print("from cache", from_cache.timestamp)
        return runs
    def _calc_robustness_score(self, df: DataFrame):
        f1_sum = 0
        weight_sum = 0
        for num_attacks, weight in NUM_ATTACK_WEIGHTS.items():
            f1_sum += df.query("`params.num_attacks` == %s" % num_attacks)["metrics.ids.f1_cfa"].iloc[0] * weight
            weight_sum += weight
        return f1_sum / weight_sum


