import argparse

import pandas as pd
import plotly.express as px
from pandas import DataFrame

from tsa.cli.check import load_exp_from_parser
from tsa.cli.run import SubCommand
from tsa.experiment import IGNORE_SCENARIOS
from tsa.mlflow.experiment_name_conversion import ExperimentNameConversion, MlflowResultsCache

NUM_ATTACK_WEIGHTS = {  # TODO!
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    5: 1,
    10: 1
}


class EvalSubCommand(SubCommand):
    def __init__(self):
        super().__init__("eval", "evaluate multiple experiments for their robustness")

    def make_subparser(self, parser: argparse.ArgumentParser):
        parser.add_argument("-c", "--config", required=False, nargs="+",
                            help="Experiment config yaml file. If not set, the config is loaded from the first mlflow run.")
        parser.add_argument("--allow-unfinished", required=False, default=False, action="store_true")
        parser.add_argument("--cache", default=None, help="If set, use a local directory to cache mlflow results.")
        parser.add_argument("--query", "-q", default=None, help="Filter result dataframes using a pandas query")
        parser.add_argument("--plot-y", default="metrics.ids.f1_cfa")

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
            runs = runs.loc[~runs["params.dataloader.scenario"].isin(IGNORE_SCENARIOS)]
            runs["params.num_attacks"] = pd.to_numeric(runs["params.num_attacks"])
            if args.query is not None:
                runs = runs.query(args.query, engine="python")
            aggregated: DataFrame = runs.filter(items=["params.num_attacks",
                                                       "metrics.ids.f1_cfa",
                                                       "metrics.ids.precision_with_cfa",
                                                       "metrics.ids.detection_rate",
                                                       "metrics.ids.consecutive_false_positives_normal",
                                                       "metrics.ids.consecutive_false_positives_exploits",
                                                       "experiment_name",
                                                       "metrics.ids.recall"]
                                                ).groupby("params.num_attacks").mean(numeric_only=False).reset_index()
            robustness_values[name] = self._calc_robustness_scores(aggregated)
            aggregated.sort_values(by=["params.num_attacks"], inplace=True)
            aggregated["experiment_name"] = name
            dfs.append(aggregated)
            # print(group_by)
            # print("===========================\n")
        merged = pd.concat(dfs)
        fig = px.line(merged,
                      x="params.num_attacks",
                      y=args.plot_y,
                      color="experiment_name",
                      line_dash="experiment_name",
                      line_dash_sequence=["dot"],
                      markers=True)
        fig.show()
        robustness_scores_df = pd.DataFrame(robustness_values)
        print(robustness_scores_df)

    def _get_results(self, checker, exp_name, cache, allow_unfinished):
        if cache is None:
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

    def _calc_robustness_scores(self, df: DataFrame, num_attack_col="params.num_attacks"):
        if num_attack_col not in df.columns:
            raise ValueError("%s is not a column in the dataframe" % num_attack_col)

        num_attack_values = pd.unique(df["params.num_attacks"])
        metrics = [c for c in df.columns if c != num_attack_col]
        metrics_sums = {m: 0 for m in metrics}
        weight_sum = 0
        if 0 not in num_attack_values:
            raise ValueError("0 must be in num_attack_values")
        for num_attacks in num_attack_values:
            if num_attacks == 0:
                continue
            for m in metrics:
                metrics_sums[m] += df.query(f"`{num_attack_col}` == {num_attacks}")[m].iloc[0]
            weight_sum += 1

        metrics_means = {m: metrics_sums[m] / weight_sum for m in metrics}

        clean_data_metrics = {m: df.query(f"`{num_attack_col}` == 0")[m].iloc[0] for m in metrics}
        relative_robustness_score = {m: metrics_means[m] / clean_data_metrics[m] for m in metrics}
        auc_metrics_w0 = {m: self._calc_auc_metrics(df, num_attack_col, m) for m in metrics}
        auc_metrics_geq1 = {m: self._calc_auc_metrics(df.query(f"`{num_attack_col}` != 0"), num_attack_col, m) for m in
                            metrics}

        # write the scores into a single dict
        scores = {}
        for prefix, scores_dict in [("mean", metrics_means), ("relative", relative_robustness_score),
                                    ("auc_w0", auc_metrics_w0), ("auc_geq1", auc_metrics_geq1)]:
            for key, metric_value in scores_dict.items():
                scores[f"{prefix}.{key}"] = metric_value
        return scores

    def _calc_auc_metrics(self, df, x_col_name, metric_name):
        X = sorted(pd.unique(df[x_col_name]))
        Y = [df.query(f"`{x_col_name}` == {x}")[metric_name].iloc[0] for x in X]
        return calc_area_under_curve(X, Y)

def calc_area_under_curve(X, Y):
    if len(X) != len(Y):
        raise ValueError("uneven length of X and Y: %s != %s" % (len(X), len(Y)))
    sum = 0
    for k in range(len(X) - 1):
        if X[k] > X[k+1]:
            raise ValueError("X is not sorted! X[%s] = %s > X[%s] = %s" % (k, X[k], k+1, X[k+1]))
        sum += (Y[k] + Y[k + 1]) / 2 * (X[k + 1] - X[k])
    return sum / (X[-1] - X[0])
