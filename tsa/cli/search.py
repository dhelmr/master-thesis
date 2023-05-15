import argparse
import itertools
import os.path
import sys
from copy import deepcopy

import yaml

from tsa.cli.run import SubCommand
from tsa.utils import access_cfg


class SearchSubCommand(SubCommand):
    def __init__(self):
        super().__init__("search", "run parameter search")

    def make_subparser(self, parser: argparse.ArgumentParser):
        parser.add_argument("-c", "--config", required=True, help="Experiment Search config yaml file.")
        parser.add_argument("-o", "--out-dir", required=False, help="Output directory.")
        parser.add_argument("--only", type=int, required=False)

    def exec(self, args, parser):
        outdir = args.out_dir
        if outdir is not None:
            if not os.path.exists(outdir):
                print("%s does not exist" % outdir)
                sys.exit(1)
            if not os.path.isdir(outdir):
                print("%s is not directory." % outdir)
                sys.exit(1)

        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        base_exp = access_cfg(cfg, "experiment")
        mode = access_cfg(cfg, "mode")
        prefix = access_cfg(cfg, "name_prefix", default="experiment")
        if mode == "grid":
            search_space = access_cfg(cfg, "search_space")
            search_keys = list(search_space.keys())
            values = search_space.values()

            for exp_i, selected_values in enumerate(itertools.product(*values)):
                if args.only is not None and exp_i != int(args.only):
                    continue
                exp = deepcopy(base_exp)
                for i, value in enumerate(selected_values):
                    key_name = search_keys[i]
                    modify_cfg(exp, key_name, value)
                if outdir is not None:
                    filename = os.path.join(outdir, "%s-%s.experiment" % (prefix, exp_i))
                    with open(filename, "w") as f:
                        print(filename)
                        yaml.dump(exp, f)
                else:
                    print(yaml.dump(exp))

        else:
            raise ValueError("Unknown mode: %s" % mode)

def modify_cfg(cfg, key, new_value):
    splitted = key.split(".")
    for i in range(len(splitted)):
        if splitted[i].isdigit():
            splitted[i] = int(splitted[i])
    parent_key = splitted[:-1]
    parent_obj = access_cfg(cfg, *parent_key, required=True)
    parent_obj[splitted[-1]] = new_value
