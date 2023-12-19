from argparse import ArgumentParser

import mlflow
import pandas as pd
from tqdm import tqdm

from tsa.cli.run import SubCommand
from tsa.mlflow.experiment_name_conversion import ExperimentNameConversion

METRIC_KEYS = {
    "metrics.ids.f1_cfa": "f1_cfa",
    "metrics.ids.precision_with_cfa": "precision_with_cfa",
    "metrics.ids.detection_rate": "detection_rate",
}


class TSACombineSubCommand(SubCommand):
    def __init__(self):
        super().__init__(
            "tsa-combine",
            "combine downloaded training set statistics and performance measures",
        )
        self.metric_keys = METRIC_KEYS

    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument(
            "--experiment",
            "-e",
            required=False,
            help="Sets the mlflow experiment name (for the performance measure experiments). If not set, it will be inferred from the config path.",
            type=str,
        )
        parser.add_argument(
            "-c", "--config", required=False, help="Experiment config yaml file."
        )
        parser.add_argument(
            "--statistics-csv",
            required=True,
            help="csv file with training set statistics (downloaded with tsa-dl subcommand)",
        )
        parser.add_argument(
            "-o", "--output", required=True, help="Output csv with combined results"
        )

    def exec(self, args, parser, unknown_args):
        statistics_df = pd.read_csv(args.statistics_csv)
        performance_df = self._dl_performance(args)
        performance_df["metrics.dataloader.training_syscalls"] = performance_df[
            "metrics.dataloader.training_syscalls"
        ].apply(self.map_n_syscalls)
        performance_df["params.dataloader.num_attacks"] = performance_df[
            "params.dataloader.num_attacks"
        ].apply(self.map_n_syscalls)
        print(performance_df.columns)
        combined = self.map_performance(statistics_df, performance_df)
        combined.to_csv(args.output, index=False)

    def map_performance(self, statistics_df, performance_df):
        data = []
        for _, r in tqdm(statistics_df.iterrows(), total=len(statistics_df)):
            row_dict = r.to_dict()
            added_one = False
            for metric, new_name in self.metric_keys.items():
                metric_val = self.get_performance(
                    performance_df,
                    r["syscalls"],
                    r["dataloader.scenario"],
                    r["dataloader.num_attacks"],
                    metric,
                )
                if metric_val is None:
                    continue
                row_dict[new_name] = metric_val
                print(new_name, metric_val)
                added_one = True
            if added_one:
                data.append(row_dict)
        return pd.DataFrame(data)

    def _dl_performance(self, args) -> pd.DataFrame:
        if args.experiment is not None:
            experiment_name = args.experiment
        else:
            converter = ExperimentNameConversion()
            experiment_name = converter.infer_exp_name(args.config)
        runs = mlflow.search_runs(experiment_names=[experiment_name])
        return runs

    def map_n_syscalls(self, x):
        try:
            return int(x)
        except ValueError as e:
            print(e)
            return -1

    def get_performance(
        self, df, syscall_num: int, scenario: str, num_attacks: int, metric_key
    ):
        selected = df.loc[
            (df["metrics.dataloader.training_syscalls"] == int(syscall_num))
            & (df["params.dataloader.scenario"] == scenario)
            & (df["params.dataloader.num_attacks"] == int(num_attacks))
        ][metric_key]
        if len(selected) >= 1:
            aslist = selected.tolist()
            asset = set(aslist)
            if len(asset) > 1:
                raise ValueError(asset.__str__())
            return aslist[0]
        return None
