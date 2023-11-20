import argparse
import sys

from tsa.cli.check import CheckSubCommand
from tsa.cli.eval import EvalSubCommand
from tsa.cli.run import RunSubCommand
from tsa.cli.search import SearchSubCommand
from tsa.cli.tsa_add_suffix import TSAAddSuffixSubCommand
from tsa.cli.tsa_augment import TSAAugmentSubCommand
from tsa.cli.tsa_combine import TSACombineSubCommand
from tsa.cli.tsa_concat import TSAConcatSubCommand
from tsa.cli.tsa_correlate import TSACorrelateSubCommand
from tsa.cli.tsa_cv import TSACrossValidateSubCommand
from tsa.cli.tsa_dl import TSADownloaderSubCommand
from tsa.cli.tsa_eval_fs import TSAEvalFsSubCommand
from tsa.cli.tsa_fs import TSAFsSubCommand
from tsa.cli.tsa_ngram_auc import TSANgramAUCSubCommand
from tsa.cli.tsa_ruleminer import TSARuleMinerSubCommand
from tsa.cli.tsa_stats import TSAStatsSubCommand

commands = [RunSubCommand(),
            CheckSubCommand(),
            TSADownloaderSubCommand(),
            TSACrossValidateSubCommand(),
            TSACombineSubCommand(),
            SearchSubCommand(),
            EvalSubCommand(),
            TSAAugmentSubCommand(),
            TSAFsSubCommand(),
            TSARuleMinerSubCommand(),
            TSACorrelateSubCommand(),
            TSAAddSuffixSubCommand(),
            TSAStatsSubCommand(),
            TSANgramAUCSubCommand(),
            TSAConcatSubCommand(),
            TSAEvalFsSubCommand()
            ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    for comm in commands:
        sparser = subparsers.add_parser(comm.name, description=comm.desc)
        comm.make_subparser(sparser)

    args, unknown_args = parser.parse_known_args()
    for comm in commands:
        if comm.name == args.command:
            if not comm.expect_unknown_args and unknown_args is not None and len(unknown_args) > 0:
                raise RuntimeError("Unexpected arguments: %s" % unknown_args)
            comm.exec(args, parser, unknown_args)
            sys.exit(0)

    parser.print_help()
    if args.command is not None:
        print("Unknown subcommand: ", args.command)
    sys.exit(1)

