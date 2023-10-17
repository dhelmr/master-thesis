"""
Example execution of LIDS Framework
"""
import abc
import argparse
import os
from dataclasses import dataclass
import random

import mlflow
import numpy as np
import torch
import yaml
from mlflow import MlflowClient
from typing import List

from tsa.analysis.analyse_experiment import AnalysisExperiment
from tsa.experiment import Experiment
from tsa.experiment_checker import ExperimentChecker
from tsa.unsupervised.evaluation import UnsupervisedExperiment
from tsa.utils import access_cfg

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def convert_mlflow_dict(nested_dict: dict):
    mlflow_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            for subkey, subvalue in convert_mlflow_dict(value).items():
                mlflow_dict[f"{key}.{subkey}"] = subvalue
        else:
            mlflow_dict[key] = str(value)
    for key, value in dict(mlflow_dict).items():
        if len(value) > 500:
            print("Skip", key, "because it exceeds mlflow length limit")
            del mlflow_dict[key]
    return mlflow_dict


def last_successful_run_id(mlflow_client: MlflowClient, experiment_name: str) -> str:
    exp = mlflow_client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError("Experiment with name '%s' not found." % experiment_name)
    for r in mlflow_client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"]):
        if r.info.status == "FINISHED":
            return r.info.run_id
    return None


def last_running_run_id(mlflow_client: MlflowClient, experiment_name: str) -> str:
    exp = mlflow_client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError("Experiment with name '%s' not found." % experiment_name)
    for r in mlflow_client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"]):
        if r.info.status == "RUNNING":
            return r.info.run_id
    return None

class SubCommand:

    def __init__(self, name, desc, expect_unknown_args=False):
        self.name = name
        self.desc = desc
        self.expect_unknown_args = expect_unknown_args
    @abc.abstractmethod
    def make_subparser(self, parser: argparse.ArgumentParser):
        raise NotImplementedError()

    @abc.abstractmethod
    def exec(self, args, parser, unknown_args):
        raise NotImplementedError()



class RunSubCommand(SubCommand):
    def __init__(self):
        super().__init__("run", "run experiment")
    def make_subparser(self, parser):
        parser.add_argument("-c", "--config", required=True, help="Experiment config yaml file.")
        parser.add_argument("--start-at", "-s", required=False, help="Start at iteration", type=int, default=0)
        parser.add_argument("--continue", required=False, dest="continue_experiment", default=None,
                            help="Continue from last successfully 'finished' or currently 'running' run of the experiment",
                            choices=["finished", "running", "random"])
        parser.add_argument("--dry-run", action="store_true", default=False,
                            help="If set, only attack mixin datasets will be created, and no Anomaly Detection is done. Useful for debugging.")
        parser.add_argument("--experiment", "-e", required=False, help="Sets the mlflow experiment ID", type=str)
        parser.add_argument("--number-runs", required=False, help="Only iterate for specific number of runs", type=int,
                            default=None)

    def exec(self, args, parser, unknown_args):
        mlflow_client = MlflowClient()
        if args.experiment is None and args.continue_experiment is True:
            print("If --continue-experiment is set, an experiment must be specified with -e.")
            parser.print_help()
            exit(1)
        if args.experiment is not None and not args.continue_experiment:
            mlflow.set_experiment(args.experiment)

        experiment = make_experiment_from_path(args.config, mlflow_client, args.experiment)

        experiment_start_args = {
            "num_runs": args.number_runs,
            "dry_run": args.dry_run
        }
        if args.continue_experiment is not None:
            start_at = get_next_iteration(mlflow_client, experiment, args.continue_experiment)
            if start_at is None:
                return
            print("Start at iteration", start_at)
        else:
            start_at = args.start_at
        experiment.start(start_at, **experiment_start_args)

def get_next_iteration(mlflow_client, experiment, mode):

    if mode == "finished":
        run_id = last_successful_run_id(mlflow_client, experiment.mlflow_name)
    elif mode == "running":
        run_id = last_running_run_id(mlflow_client, experiment.mlflow_name)
        if run_id is None:
            print("No running run found; find last successfully finished run.")
            run_id = last_successful_run_id(mlflow_client, experiment.mlflow_name)
    elif mode == "random":
        checker = ExperimentChecker(experiment)
        if checker.exists_in_mlflow():
            stats = checker.stats()
            missing_not_running = stats.missing_but_not_running()
            if len(missing_not_running) == 0:
                print("Experiment has no missing runs")
                return None
            missing_runs = missing_not_running
        else:
            missing_runs = experiment.run_configurations()
        r = random.Random(None)
        next_run = r.choice(missing_runs)
        return next_run.iteration
    else:
        raise RuntimeError("continue_experiment has unexpected value: %s" % mode)
    if run_id is not None:
        print("Continue from run %s" % run_id)
        mlflow.set_experiment(experiment.mlflow_name)
        run = mlflow_client.get_run(run_id)
        iteration = int(run.data.params.get("iteration"))
        return iteration + 1

def make_experiment_from_path(path, mlflow_client, name):
    with open(path) as f:
        parameter_config = yaml.safe_load(f)
    if access_cfg(parameter_config, "id", required=False) is None:
        parameter_config["id"] = path.split(os.path.sep)[-1]
    return make_experiment(parameter_config, mlflow_client, name)

def get_exp_config_from_mlflow(mlflow_client: MlflowClient, exp_id):
    runs = mlflow_client.search_runs(experiment_ids=[exp_id])
    for r in runs:
        return mlflow.artifacts.load_dict(r.info.artifact_uri + "/config.json")
    raise RuntimeError("No run found for experiment id %s" % exp_id)
def make_experiment_from_mlflow(mlflow_client: MlflowClient, mlflow_exp_name):
    exp = mlflow_client.get_experiment_by_name(mlflow_exp_name)
    config = get_exp_config_from_mlflow(mlflow_client, exp.experiment_id)
    return make_experiment(config, mlflow_client, mlflow_exp_name)




def make_experiment(parameter_config, mlflow_client, name):
    exp_mode = access_cfg(parameter_config, "mode", default="normal")
    if exp_mode == "normal":
        experiment = Experiment(parameter_config, mlflow=mlflow_client, mlflow_name=name)
    elif exp_mode == "unsupervised":
        experiment = UnsupervisedExperiment(parameter_config, mlflow=mlflow_client, mlflow_name=name)
    elif exp_mode == "analysis":
        experiment = AnalysisExperiment(parameter_config, mlflow=mlflow_client, mlflow_name=name)
    else:
        raise ValueError("Unexpected experiment mode: %s" % exp_mode)
    return experiment
