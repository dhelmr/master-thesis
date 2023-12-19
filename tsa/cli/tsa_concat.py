from argparse import ArgumentParser

import pandas

from tsa.cli.run import SubCommand


class TSAConcatSubCommand(SubCommand):
    def __init__(self):
        super().__init__("tsa-concat", "concat training set characteristics data files")

    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument(
            "-i",
            "--input",
            required=True,
            help="input data files (training set statistics -> performance)",
            nargs="+",
        )
        parser.add_argument("-o", "--output", help="Output csv file")
        parser.add_argument(
            "--common",
            help="Common column names",
            nargs="+",
            default=["f1_cfa", "detection_rate", "precision_with_cfa"],
        )
        parser.add_argument("--skip", nargs="+", default=[])

    def exec(self, args, parser, unknown_args):
        merge_keys = [
            "scenario",
            "permutation_i",
            "iteration",
            "is_last",
            "syscalls",
            "num_attacks",
        ]
        for key in args.common:
            if key not in merge_keys:
                merge_keys.append(key)
        merged_df = None
        for i_file in args.input:
            df = pandas.read_csv(i_file)
            drop_cols = []
            for c in args.skip + ["parameter_cfg_id", "run_id"]:
                if c in df.columns and c not in drop_cols:
                    drop_cols.append(c)
            df.drop(columns=drop_cols, inplace=True)
            if merged_df is None:
                merged_df = df
                continue
            cols_to_use = df.columns.difference(merged_df.columns).to_list()
            cols_to_use += merge_keys
            merged_df = pandas.merge(
                merged_df, df[cols_to_use], on=merge_keys, suffixes=("", "")
            )
        merged_df.to_csv(args.output)
