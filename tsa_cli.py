import argparse
import sys

from tsa.cli.check import CheckSubCommand
from tsa.cli.run import RunSubCommand
from tsa.cli.search import SearchSubCommand
from tsa.cli.tsa import TSASubCommand

commands = [RunSubCommand(), CheckSubCommand(), TSASubCommand(), SearchSubCommand()]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    for comm in commands:
        sparser = subparsers.add_parser(comm.name, description=comm.desc)
        comm.make_subparser(sparser)
    args = parser.parse_args()
    for comm in commands:
        if comm.name == args.command:
            comm.exec(args, parser)
            sys.exit(0)

    parser.print_help()
    print("Unknown subcommand: ", args.command)
    sys.exit(1)

