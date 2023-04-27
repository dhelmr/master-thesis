"""
Example execution of LIDS Framework
"""
import argparse
import math
import os
import pickle
import sys
import datetime
import tempfile
import uuid
from dataclasses import dataclass
from pprint import pprint
import random
from shutil import copyfile, copytree

import mlflow
import numpy as np
import pandas
import pandas as pd
import torch
import yaml
from mlflow import MlflowClient
from typing import List

from mlflow.entities import RunStatus

from algorithms.decision_engines.scg import SystemCallGraph
from algorithms.decision_engines.som import Som
from algorithms.decision_engines.stide import Stide
from algorithms.features.impl.stream_sum import StreamSum
from algorithms.features.impl.w2v_embedding import W2VEmbedding
from algorithms.performance_measurement import Performance
from tsa.analyse import TrainingSetAnalyser
from tsa.confusion_matrix import ConfusionMatrix
from dataloader.dataloader_factory import dataloader_factory

from dataloader.direction import Direction

from algorithms.ids import IDS

from algorithms.features.impl.max_score_threshold import MaxScoreThreshold
from algorithms.features.impl.one_hot_encoding import OneHotEncoding
from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.syscall_name import SyscallName
from algorithms.features.impl.ngram import Ngram
from algorithms.decision_engines.ae import AE
from tsa.dataloader_2019 import ContaminatedDataLoader2019
from tsa.dataloader_2021 import ContaminatedDataLoader2021
from tsa.preprocessing import OutlierDetector, LOF, MixedModelOutlierDetector

try:
    LID_DS_BASE_PATH = os.environ['LID_DS_BASE']
except KeyError as exc:
    raise ValueError("No LID-DS Base Path given."
                     "Please specify as argument or set Environment Variable "
                     "$LID_DS_BASE") from exc


@dataclass
class AttackMixinConfig:
    dataset_version: str  # TODO use enum
    num_attacks: int
    scenario_name: str
    scenario_path: str
    attack_names: List[str]

    def to_dict(self):
        return self.__dict__


DECISION_ENGINES = {de.__name__: de for de in [AE, Stide, Som, SystemCallGraph]}
PREPROCESSORS = {
    "MixedModelOD": MixedModelOutlierDetector,
    "LOF": LOF
}

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class Experiment:
    def __init__(self, parameters, mlflow: MlflowClient):
        self._tmp_results_df = None
        self.parameters = parameters
        self.mlflow = mlflow
        self.scenarios = self._get_param("scenarios", exp_type=list)

    def start(self, start_at=0, dry_run=False):
        max_attacks = self._get_param("attack_mixin", "max_attacks", exp_type=int)
        dataloader_config = self._get_param("attack_mixin", "dataloader", exp_type=dict)
        i = -1
        for scenario in self.scenarios:
            lid_ds_version, scenario_name = scenario.split("/")
            dataloader_class = self._get_dataloader_cls(lid_ds_version)
            scenario_path = f"{LID_DS_BASE_PATH}/{lid_ds_version}/{scenario_name}"
            for num_attacks in range(max_attacks + 1):
                dataloader = dataloader_class(scenario_path, num_attacks=num_attacks, direction=Direction.BOTH,
                                              **dataloader_config)
                i = i + 1
                if i < start_at:
                    print("Skip", dataloader.__dict__)
                    continue
                if dry_run:
                    print(i, "Dry Run: ", dataloader.__dict__)
                    continue
                with mlflow.start_run() as run:
                    mlflow.log_params(convert_mlflow_dict(dataloader.cfg_dict()))
                    mlflow.log_params(convert_mlflow_dict(self.parameters))
                    mlflow.log_params(convert_mlflow_dict({"iteration": i}))
                    additional_params, results, ids = self.train_test(dataloader)
                    self.log_artifacts(ids)
                    mlflow.log_params(convert_mlflow_dict(additional_params))
                    for metric_key, value in convert_mlflow_dict(results).items():
                        try:
                            value = float(value)
                        except ValueError:
                            print("Skip metric", metric_key)
                            continue
                        mlflow.log_metric(metric_key, value)

    def _get_dataloader_cls(self, lid_ds_version):
        if lid_ds_version == "LID-DS-2019":
            return ContaminatedDataLoader2019
        if lid_ds_version == "LID-DS-2021":
            return ContaminatedDataLoader2021
        else:
            raise ValueError("%s is not supported." % lid_ds_version)

    def make_feature_extractor(self, cfg_prefix=[]):
        syscall_name = SyscallName()
        feature_name = self._get_param(*cfg_prefix, "features", "name")
        if feature_name == "OneHotEncodingNgram":
            embedding = OneHotEncoding(syscall_name)
        elif feature_name == "IntEmbeddingNgram":
            embedding = IntEmbedding(syscall_name)
        elif feature_name == "W2VEmbedding":
            embedding = W2VEmbedding(syscall_name,
                                     window_size=self._get_param(*cfg_prefix, "features", "window_size"),
                                     vector_size=self._get_param(*cfg_prefix, "features", "vector_size"),
                                     epochs=self._get_param(*cfg_prefix, "features", "epochs")
                                     )
        else:
            raise ValueError("%s is not a valid name" % feature_name)
        thread_aware = self._get_param(*cfg_prefix, "features", "thread_aware", default=True)
        n_gram_length = self._get_param(*cfg_prefix, "features", "n_gram_length", default=7)
        features = Ngram([embedding], thread_aware, n_gram_length)
        return features

    def train_test(self, dataloader):
        # just load < closing system calls for this example

        DecisionEngineClass = DECISION_ENGINES[self.parameters["decision_engine"]["name"]]

        features = self.make_feature_extractor()
        # TODO: "Pipeline" class/obj
        analyser = TrainingSetAnalyser(features)
        analysers = [analyser]
        for i, _ in enumerate(self._get_param("preprocessing", default=[])):
            cfg_prefix = ["preprocessing", i]
            preprocessor_name = self._get_param(*cfg_prefix, "name")
            if preprocessor_name not in PREPROCESSORS:
                raise ValueError(f"{preprocessor_name} is not a valid preprocessor.")
            args = self._get_param(*cfg_prefix, "args", default={})
            train_features = None
            if self._exists_param(*cfg_prefix, "features"):
                train_features = self.make_feature_extractor(cfg_prefix)
            features = PREPROCESSORS[preprocessor_name](analyser, train_features=train_features, **args)
            analyser = TrainingSetAnalyser(features)
            analysers.append(analyser)

        decision_engine_args = self._get_param("decision_engine", "args", default={}, exp_type=dict)
        decision_engine = DecisionEngineClass(analyser, **decision_engine_args)
        if DecisionEngineClass in {DECISION_ENGINES["Stide"], DECISION_ENGINES["SystemCallGraph"]}:
            window_length = self._get_param("decision_engine", "streaming_window_length", default=1000, exp_type=int)
            decision_engine = StreamSum(decision_engine, False, window_length, False)
        # decider threshold
        decider_1 = MaxScoreThreshold(decision_engine)
        ### the IDS
        ids = IDS(data_loader=dataloader,
                  resulting_building_block=decider_1,
                  create_alarms=True,
                  plot_switch=False)

        print("at evaluation:")
        performance = ids.detect()

        results = self.calc_extended_results(performance)
        additional_parameters = {
            f"tsa{i + 1}": analyser.get_analyse_result() for i, analyser in enumerate(analysers)
        }
        additional_parameters["config"] = ids.get_config_tree_links(),
        return additional_parameters, results, ids

    def calc_extended_results(self, performance: Performance):
        results = performance.get_results()
        cm = ConfusionMatrix(tn=results["true_negatives"], fp=results["false_positives"], tp=results["true_positives"],
                             fn=results["false_negatives"])
        metrics = cm.calc_unweighted_measurements()
        return {"ids": results, "cm": metrics}

    def _exists_param(self, *keys) -> bool:
        val = self._get_param(*keys, default=None, required=False)
        return val is not None

    def _get_param(self, *keys, default=None, required=True, exp_type=None):
        if default is not None:
            required = False
        cur_obj = self.parameters
        cur_key = ""
        for key in keys:
            if not isinstance(cur_obj, dict) and not isinstance(cur_obj, list):
                raise ValueError("Parameter '%s' is not a dict or list." % cur_key)
            if isinstance(cur_obj, dict) and key not in cur_obj:
                if not required:
                    return default
                else:
                    raise ValueError("Cannot find parameter for key %s at %s" % (key, cur_key))
            cur_obj = cur_obj[key]
            cur_key = "%s.%s" % (cur_key, key)
        if exp_type is not None and not isinstance(cur_obj, exp_type):
            raise ValueError("Parameter %s is not of expected type %s" % (cur_key, exp_type))
        return cur_obj

    def log_artifacts(self, ids):
        mlflow.log_dict(self.parameters, "config.json")

    @staticmethod
    def continue_run(mlflow_client: MlflowClient, run_id: str, dry_run=False):
        run = mlflow_client.get_run(run_id)
        artifact_uri = run.info.artifact_uri
        config_json = mlflow.artifacts.load_dict(artifact_uri + "/config.json")
        iteration = int(run.data.params.get("iteration"))
        Experiment(config_json, mlflow_client).start(iteration + 1, dry_run=dry_run)


def split_list(l, fraction_sublist1: float):
    if fraction_sublist1 < 0 or fraction_sublist1 > 1:
        raise ValueError("Argument fraction_sublist1 must be between 0 and 1, but is: %s" % fraction_sublist1)
    size = len(l)
    split_at = math.floor(fraction_sublist1 * size)
    return l[:split_at], l[split_at:]


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
    for r in mlflow_client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC", "end_time DESC"]):
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
    if args.config is None:
        if args.continue_run is not None:
            run_id = args.continue_run
        elif args.continue_experiment:
            run_id = last_successful_run_id(mlflow_client, args.experiment)
        else:
            raise RuntimeError("Expected the run_id to be set. Abort.")
        print("Continue from run %s" % run_id)
        Experiment.continue_run(mlflow_client, run_id, args.dry_run)
    else:
        with open(args.config) as f:
            exp_parameters = yaml.safe_load(f)
        start_at = args.start_at
        Experiment(exp_parameters, mlflow=mlflow_client).start(start_at, dry_run=args.dry_run)
