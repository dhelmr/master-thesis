from argparse import ArgumentParser

from tsa.cli.run import SubCommand
from tsa.cli.tsa_cv import load_data


class TSAAddSuffixSubCommand(SubCommand):
    def __init__(self):
        super().__init__("tsa-add-suffix", "add suffix to features in csv file")

    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("-i", "--input", required=True, help="input data file")
        parser.add_argument("-o", "--output", help="Output csv file")
        parser.add_argument("--scenario-column", default="scenario")
        parser.add_argument("--features", "-f", required=False, default=None, nargs="+")
        parser.add_argument("--skip-features", nargs="+", default=[])
        parser.add_argument("--suffix", required=True)

    def exec(self, args, parser, unknown_args):
        data = load_data(
            args.input, args.scenario_column, args.features, args.skip_features
        )
        rename_cols = {c: f"{c}{args.suffix}" for c in data.feature_cols()}
        renamed_df = data.df.rename(columns=rename_cols)
        renamed_df.to_csv(args.output)
