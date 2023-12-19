import argparse
import os.path

import pandas
import pandas as pd
import plotly.express as px
from pandas import DataFrame

from tsa.cli.check import load_exp_from_parser
from tsa.cli.run import SubCommand
from tsa.mlflow.experiment_name_conversion import ExperimentNameConversion, MlflowResultsCache
from tsa.utils import md5

METRIC_LABELS = {
    "metrics.ids.f1_cfa": "mean f1-score",
    "metrics.ids.precision_with_cfa": "mean precision",
    "metrics.ids.detection_rate": "mean detection rate",
    "metrics.ids.consecutive_false_positives_normal": "mean cfp-normal",
    "metrics.ids.consecutive_false_positives_exploits": "mean cfp-exploits"
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
        parser.add_argument("--names", default=None, nargs="+", type=str)
        parser.add_argument("--artifacts-dir", default=None, type=str,
                            help="Store artifacts to a directory (created if not exists)")

    def exec(self, args, parser, unknown_args):
        if args.artifacts_dir is not None and not os.path.exists(args.artifacts_dir):
            print("Create artifacts dir", args.artifacts_dir)
            os.mkdir(args.artifacts_dir)
        if args.artifacts_dir is not None and not os.path.isdir(args.artifacts_dir):
            raise FileExistsError("Is not a directory: %s" % args.artifacts_dir)
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
        self.union_ignore_scenarios(checkers)
        robustness_values = {}
        dfs = []
        for i, checker in enumerate(checkers):
            name = converter.get_rel_exp_name(checker.experiment.mlflow_name)
            print(checker.experiment.mlflow_name)
            runs = self._get_results(checker, name, cache, args.allow_unfinished)
            runs = runs.loc[~runs["params.dataloader.scenario"].isin(checker.experiment.ignore_scenarios)]
            runs["params.num_attacks"] = pd.to_numeric(runs["params.num_attacks"])
            if args.query is not None:
                runs = runs.query(args.query, engine="python")
            aggregated: DataFrame = runs.filter(items=["params.num_attacks",
                                                       "metrics.ids.f1_cfa",
                                                       "metrics.ids.precision_with_cfa",
                                                       "metrics.ids.detection_rate",
                                                       "metrics.ids.consecutive_false_positives_normal",
                                                       "metrics.ids.consecutive_false_positives_exploits",
                                                       "experiment_name"]
                                                ).groupby("params.num_attacks").mean(numeric_only=False).reset_index()
            if args.names is not None and args.names[i] != "-":
                name = args.names[i]
            robustness_values[name] = self._calc_robustness_scores(aggregated)
            aggregated.sort_values(by=["params.num_attacks"], inplace=True)
            aggregated["Experiment"] = name

            dfs.append(aggregated)
            # print(group_by)
            # print("===========================\n")
        merged = pd.concat(dfs)
        robustness_scores_df = pd.DataFrame(robustness_values)
        print(robustness_scores_df)
        if args.artifacts_dir is not None:
            for metric in METRIC_LABELS.keys():
                image_path = os.path.join(args.artifacts_dir, f"{metric}.svg")
                self.make_metric_fig(merged, metric=metric, img_path=image_path)
            r_transposed = robustness_scores_df.transpose()
            r_transposed["experiment"] = r_transposed.index
            r_transposed.to_csv(
                path_or_buf=os.path.join(args.artifacts_dir, f"robustness-measures.csv")
            )
            self.make_robustness_fig(r_transposed,
                                     metrics=["metrics.ids.f1_cfa",
                                              "metrics.ids.precision_with_cfa",
                                              "metrics.ids.detection_rate"],
                                     img_path=os.path.join(args.artifacts_dir, f"auc.svg"),
                                     metrics_prefix="auc")
            self.make_robustness_fig(r_transposed,
                                     metrics=["metrics.ids.f1_cfa",
                                              "metrics.ids.precision_with_cfa",
                                              "metrics.ids.detection_rate"],
                                     img_path=os.path.join(args.artifacts_dir, f"relative.svg"),
                                     metrics_prefix="relative")
            self.make_robustness_fig(r_transposed,
                                     metrics=["metrics.ids.consecutive_false_positives_normal",
                                              "metrics.ids.consecutive_false_positives_exploits"],
                                     img_path=os.path.join(args.artifacts_dir, f"auc-cfp.svg"),
                                     metrics_prefix="auc")
            self.make_robustness_fig(r_transposed,
                                     metrics=["metrics.ids.consecutive_false_positives_normal",
                                              "metrics.ids.consecutive_false_positives_exploits"],
                                     img_path=os.path.join(args.artifacts_dir, f"relative-cfp.svg"),
                                     metrics_prefix="relative")

    def make_robustness_fig(self, robustness_df: pandas.DataFrame, metrics_prefix, metrics, img_path):
        robustness_df = robustness_df.drop(columns=[c for c in robustness_df.columns if c not in
                                                    [f"{metrics_prefix}.{m}" for m in metrics] + ["experiment"]])
        robustness_df = robustness_df.melt(
            id_vars=["experiment"],
            var_name="metric",
            value_name="value"
        )
        robustness_df["metric"] = robustness_df["metric"].apply(
            lambda metric: f"{metrics_prefix} {METRIC_LABELS[metric[len(metrics_prefix) + 1:]]}"
        )
        print(robustness_df)
        # for metric in ["metrics.ids.f1_cfa", "metrics.ids.detection_rate", "metrics.ids.precision_with_cfa"]:
        r_fig = px.bar(robustness_df, x="experiment", y="value", color="metric", barmode="group")
        # r_fig.update_layout(yaxis_title=f"AUC of {METRIC_LABELS[metric]}")
        r_fig.write_image(img_path)

    def make_metric_fig(self, df: pandas.DataFrame, metric: str, img_path):
        fig = px.line(df,
                      x="params.num_attacks",
                      y=metric,
                      color="Experiment",
                      line_dash="Experiment",
                      line_dash_sequence=["dot"],
                      markers=True)
        if metric not in ["metrics.ids.consecutive_false_positives_normal",
                          "metrics.ids.consecutive_false_positives_exploits"]:
            fig.update_layout(
                yaxis=dict(range=[0, 1])
            )
            fig.update_yaxes(gridwidth=0.5, griddash="dot", dtick=0.1)
        fig.update_layout(
            yaxis_title=METRIC_LABELS[metric],
            xaxis_title="Number of Attacks"
        )
        fig.write_image(img_path)

    def _get_results(self, checker, exp_name, cache, allow_unfinished):
        if cache is None:
            runs, _ = checker.get_runs_df(no_finished_check=allow_unfinished)
            return runs

        cache_key = exp_name + md5(checker.experiment.ignore_scenarios, checker.experiment.parameter_cfg)
        from_cache = cache.get_cached_result(cache_key)
        if from_cache is None:
            runs, is_finished = checker.get_runs_df(no_finished_check=allow_unfinished)
            if is_finished:
                cache.cache(cache_key, runs)
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

        # metrics_means = {m: metrics_sums[m] / weight_sum for m in metrics}

        clean_data_metrics = {m: df.query(f"`{num_attack_col}` == 0")[m].iloc[0] for m in metrics}
        auc_metrics_w0 = {m: self._calc_auc_metrics(df, num_attack_col, m) for m in metrics}
        # auc_metrics_geq1 = {m: self._calc_auc_metrics(df.query(f"`{num_attack_col}` != 0"), num_attack_col, m) for m in
        #                    metrics}
        relative_robustness_score = {m: auc_metrics_w0[m] / clean_data_metrics[m] for m in metrics}

        # write the scores into a single dict
        scores = {}
        for prefix, scores_dict in [("auc", auc_metrics_w0), ("relative", relative_robustness_score)]:
            # ("mean", auc_metrics_w0), ("auc_geq1", auc_metrics_geq1)]:
            for key, metric_value in scores_dict.items():
                scores[f"{prefix}.{key}"] = metric_value
        return scores

    def _calc_auc_metrics(self, df, x_col_name, metric_name):
        X = sorted(pd.unique(df[x_col_name]))
        Y = [df.query(f"`{x_col_name}` == {x}")[metric_name].iloc[0] for x in X]
        return calc_area_under_curve(X, Y)

    def union_ignore_scenarios(self, checkers):
        # make the ignore_scenario lists equal for all experiments to ensure a fair comparison
        all_ignore_scenarios = []
        for checker in checkers:
            for sc in checker.experiment.ignore_scenarios:
                if sc not in all_ignore_scenarios:
                    all_ignore_scenarios.append(sc)
        for checker in checkers:
            for sc in all_ignore_scenarios:
                if sc not in checker.experiment.ignore_scenarios:
                    print("[%s] Ignore scenario %s" % (checker.experiment.parameter_cfg_id, sc))
                    checker.experiment.ignore_scenarios.append(sc)


def calc_area_under_curve(X, Y):
    if len(X) != len(Y):
        raise ValueError("uneven length of X and Y: %s != %s" % (len(X), len(Y)))
    sum = 0
    for k in range(len(X) - 1):
        if X[k] > X[k + 1]:
            raise ValueError("X is not sorted! X[%s] = %s > X[%s] = %s" % (k, X[k], k + 1, X[k + 1]))
        sum += (Y[k] + Y[k + 1]) / 2 * (X[k + 1] - X[k])
    return sum / (X[-1] - X[0])
