from argparse import ArgumentParser

import pandas
import pandas as pd
from tqdm import tqdm

from tsa.cli.eval import calc_area_under_curve
from tsa.cli.run import SubCommand
from tsa.cli.tsa_cv import load_data


class TSANgramAUCSubCommand(SubCommand):
    def __init__(self):
        super().__init__(
            "tsa-ngram-auc",
            "calculate Area-Under-Curve Values for n-gram related metrics",
        )

    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument(
            "-i",
            "--input",
            required=True,
            help="input data file (training set statistics -> performance)",
        )
        parser.add_argument("--features", "-f", required=False, nargs="+", default=None)
        parser.add_argument("--ngram-size-col", required=False, default="ngram_size")
        parser.add_argument("--scenario-column", default="scenario")
        parser.add_argument("--keep-ngram-size", default=[5], nargs="+", type=int)
        parser.add_argument("--out", "-o", required=True)

    def exec(self, args, parser, unknown_args):
        if len(args.keep_ngram_size) == 0:
            raise ValueError("must at least pass one value for --keep-ngram-size")
        data = load_data(args.input, args.scenario_column, args.features)
        auc_rows = []
        for sc in tqdm(data.get_scenarios()):
            sc_data = data.get_scenario_data(sc)
            for n_syscalls in pd.unique(sc_data["syscalls"]):
                measurement = sc_data.loc[
                    sc_data["syscalls"] == n_syscalls
                ].sort_values(by=args.ngram_size_col)
                ngram_size_values = measurement[args.ngram_size_col]
                row = measurement.loc[
                    measurement[args.ngram_size_col] == args.keep_ngram_size[0]
                ].to_dict("records")[0]
                for f in data.feature_cols():
                    if f == args.ngram_size_col:
                        continue
                    Y_values = measurement[f]
                    if f == "conditional_entropy":
                        # TODO filter out NaN values dynamically
                        auc = calc_area_under_curve(
                            ngram_size_values.tolist()[1:], Y_values.tolist()[1:]
                        )
                    else:
                        auc = calc_area_under_curve(
                            ngram_size_values.tolist(), Y_values.tolist()
                        )
                    # row[f"{f}-n{args.keep_ngram_size[0]}"] = row[f]
                    for ngram_size in args.keep_ngram_size:
                        row[f"{f}@n{ngram_size}"] = measurement.loc[
                            measurement[args.ngram_size_col] == ngram_size
                        ].iloc[0][f]
                    row[f"{f}@auc"] = auc
                    # add feature for other ngram sizes

                    del row[f]
                del row[args.ngram_size_col]
                auc_rows.append(row)
        auc_df = pandas.DataFrame(auc_rows)
        auc_df.to_csv(args.out)
