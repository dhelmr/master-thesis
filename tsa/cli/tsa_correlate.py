import math
import pprint
from argparse import ArgumentParser

import pandas
import pandas as pd

from tsa.cli.run import SubCommand
from tsa.cli.tsa_cv import load_data


class TSACorrelateSubCommand(SubCommand):

    def __init__(self):
        super().__init__("tsa-correlate", "show correlations of training set statistics")
    def make_subparser(self, parser: ArgumentParser):
        parser.add_argument("-i", "--input", required=True,
                            help="input data file (training set statistics -> performance)")
        parser.add_argument("--features", "-f", required=False, nargs="+", default=None)
        parser.add_argument("--target", default="f1_cfa")
       # parser.add_argument("--threshold", default=0.8, type=float)
        parser.add_argument("--scenario-column", default="scenario")
        parser.add_argument("--only-above", default=0.2)
        parser.add_argument("-o", "--output", default=None, type=str)
        parser.add_argument("--skip-features", "-s", required=False, default=[], nargs="+")
        parser.add_argument("--scenario-mean", action="store_true", default=False, required=False, help="Calculate mean correlation over all scenarios")

    def exec(self, args, parser, unknown_args):
        data = load_data(args.input, args.scenario_column, args.features, args.skip_features)
        with pd.option_context('mode.use_inf_as_na', True):
            data.df = data.df.fillna(value=0)
        if args.scenario_mean:
            corrs = []
            for sc in data.get_scenarios():
                sc_data = data.get_scenario_data(sc)
                corr = self._calc_corr(sc_data, data.feature_cols(), args)
                corrs.append(corr)
            aggregated = pd.DataFrame(corrs)
            corr = aggregated.mean()
        else:
            corr = self._calc_corr(data.df, data.feature_cols(), args)
        corr.sort_values(inplace=True, key=lambda x: abs(x))
        as_list = list(zip(corr.index, corr))
        if args.output is not None:
            df = pandas.DataFrame(as_list, columns=["feature", "corr_coeff"])
            df.to_csv(args.output, index=False)
        pprint.pprint(as_list)

    def _calc_corr(self, df, feature_cols, args):
        corr = df[feature_cols].corrwith(df[args.target])
        #corr = corr.apply(lambda x: abs(x))
        corr.apply(lambda x: math.nan if x < args.only_above else x)
        corr.dropna(inplace=True)

        return corr
