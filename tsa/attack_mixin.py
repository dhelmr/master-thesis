"""
Example execution of LIDS Framework
"""
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

from tsa.experiment import Experiment
from tsa.unsupervised import UnsupervisedExperiment
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
    raise RuntimeError("Could not find any finished run to continue from in the experiment %s." % experiment_name)


def main():
    mlflow_client = MlflowClient()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=False, help="Experiment config yaml file.")
    parser.add_argument("--start-at", "-s", required=False, help="Start at iteration", type=int, default=0)
    parser.add_argument("--continue-run", "-r", required=False, help="Continue run id from mlflow", type=str)
    parser.add_argument("--continue", required=False, dest="continue_experiment",
                        help="Continue from last successful run of the experiment", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="If set, only attack mixin datasets will be created, and no Anomaly Detection is done. Useful for debugging.")
    parser.add_argument("--experiment", "-e", required=False, help="Sets the mlflow experiment ID", type=str)
    parser.add_argument("--number-runs", required=False, help="Only iterate for specific number of runs", type=int,
                        default=None)
    args = parser.parse_args()
    if args.config is None and args.continue_run is None and args.continue_experiment is False:
        print("Either --config or --continue-run or --continue must be set")
        parser.print_help()
        exit(1)
    if args.config is not None and args.continue_run is not None:
        print("Only one of --config and --continue-run can be set")
        parser.print_help()
        exit(1)
    if args.config is not None and args.continue_experiment is True:
        print("Only one of --config and --continue can be set")
        parser.print_help()
        exit(1)
    if args.experiment is None and args.continue_experiment is True:
        print("If --continue-experiment is set, an experiment must be specified with -e.")
        parser.print_help()
        exit(1)
    if args.experiment is not None and not args.continue_experiment:
        mlflow.set_experiment(args.experiment)
    experiment_start_args = {
        "num_runs": args.number_runs,
        "dry_run": args.dry_run
    }
    if args.config is None:
        if args.continue_run is not None:
            run_id = args.continue_run
        elif args.continue_experiment:
            run_id = last_successful_run_id(mlflow_client, args.experiment)
        else:
            raise RuntimeError("Expected the run_id to be set. Abort.")
        print("Continue from run %s" % run_id)
        mlflow.set_experiment(args.experiment)
        Experiment.continue_run(mlflow_client, run_id, **experiment_start_args)
    else:
        with open(args.config) as f:
            exp_parameters = yaml.safe_load(f)
        exp_mode = access_cfg(exp_parameters, "mode", default="normal")
        if exp_mode == "normal":
            experiment = Experiment(exp_parameters, mlflow=mlflow_client)
        elif exp_mode == "unsupervised":
            experiment = UnsupervisedExperiment(exp_parameters, mlflow=mlflow_client)
        else:
            raise ValueError("Unexpected experiment mode: %s" % exp_mode)
        experiment.start(args.start_at, **experiment_start_args)


