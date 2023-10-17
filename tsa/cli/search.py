import argparse
import itertools
import os.path
import sys
from copy import deepcopy
from typing import List, Optional

import yaml
from mlflow import MlflowClient

from tsa.cli.run import SubCommand, make_experiment, get_next_iteration
from tsa.experiment import Experiment
from tsa.experiment_checker import ExperimentChecker
from tsa.utils import access_cfg


class SearchSubCommand(SubCommand):
    def __init__(self):
        super().__init__("search", "run parameter search")

    def make_subparser(self, parser: argparse.ArgumentParser):
        parser.add_argument("-c", "--config", required=True, help="Experiment Search config yaml file.")
        parser.add_argument("--experiment", "-e", required=False, help="Sets the mlflow experiment ID", type=str)

    def exec(self, args, parser, unknown_args):
        #outdir = args.out_dir
        #if outdir is not None:
        #    if not os.path.exists(outdir):
        #        print("%s does not exist" % outdir)
        #        sys.exit(1)
        #    if not os.path.isdir(outdir):
        #        print("%s is not directory." % outdir)
        #        sys.exit(1)

        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        search = ParameterSearch(cfg)
        mlflow = MlflowClient()
        next_experiment = search.get_next(mlflow, args.experiment)
        if next_experiment is None:
            print("No parameter configurations to continue")
            return

        experiment_start_args = {
            "num_runs": 1,
            "dry_run": False
        }
        start_at = get_next_iteration(mlflow, next_experiment, mode="random")
        print("Start at iteration", start_at)

        next_experiment.start(start_at, **experiment_start_args)


class ParameterSearch:

    def __init__(self, cfg):
        self._cfg = cfg

    def iter_parameter_cfgs(self):
        mode = access_cfg(self._cfg, "mode")
        if mode == "grid":
            for exp in self.yield_grid_experiments():
                yield exp
        else:
            raise ValueError("Unknown search mode: %s" % mode)

    def iterate_exp_checkers(self, mlflow_client, mlflow_exp_name):
        for parameters in self.iter_parameter_cfgs():
            experiment = make_experiment(parameters, mlflow_client, mlflow_exp_name)
            checker = ExperimentChecker(experiment)
            yield checker
    def get_next(self, mlflow_client, mlflow_exp_name) -> Optional[Experiment]:
        for checker in self.iterate_exp_checkers(mlflow_client, mlflow_exp_name):
            experiment = checker.experiment
            if not checker.exists_in_mlflow():
                return experiment
            stats = checker.stats()
            if stats.is_finished():
                print("Experiment for parameter config %s is finished." % experiment.parameter_cfg_id)
            elif len(stats.missing_runs) == len(stats.missing_runs_but_running):
                print("All runs of experiment %s are still running, continue to next." % experiment.parameter_cfg_id)
                continue # TODO change for other modes than grid mode (if next parameter cfg depends on previous ones)
            else:
                return experiment

        return None

    def yield_grid_experiments(self):
        prefix = access_cfg(self._cfg, "name_prefix", default="experiment")
        base_exp = access_cfg(self._cfg, "experiment")
        search_space = access_cfg(self._cfg, "search_space")
        search_keys = list(search_space.keys())
        values = search_space.values()

        for exp_i, selected_values in enumerate(itertools.product(*values)):
            exp = deepcopy(base_exp)
            for i, value in enumerate(selected_values):
                key_name = search_keys[i]
                modify_cfg(exp, key_name, value)
            exp["id"] = "%s-%s" % (prefix, exp_i)
            yield exp


def modify_cfg(cfg, key, new_value):
    splitted = key.split(".")
    for i in range(len(splitted)):
        if splitted[i].isdigit():
            splitted[i] = int(splitted[i])
    parent_key = splitted[:-1]
    parent_obj = access_cfg(cfg, *parent_key, required=True)
    parent_obj[splitted[-1]] = new_value
