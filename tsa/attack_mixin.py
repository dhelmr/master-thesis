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
from pprint import pprint
from shutil import copyfile, copytree

import mlflow
import pandas
import pandas as pd
import yaml
from mlflow import MlflowClient

from algorithms.decision_engines.som import Som
from algorithms.decision_engines.stide import Stide
from algorithms.features.impl.stream_sum import StreamSum
from algorithms.features.impl.w2v_embedding import W2VEmbedding
from algorithms.performance_measurement import Performance
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

try:
    LID_DS_BASE_PATH = os.environ['LID_DS_BASE']
except KeyError as exc:
    raise ValueError("No LID-DS Base Path given."
                     "Please specify as argument or set Environment Variable "
                     "$LID_DS_BASE") from exc
LID_DS_VERSION = "LID-DS-2021"


class AttackMixin:
    def __init__(self, scenario_path: str, tmp_dir: str, step_size: int, max_attack_fraction: float):
        self.scenario_path = scenario_path
        self.tmp_dir = tmp_dir
        self.step_size = step_size
        self.max_attack_fraction = max_attack_fraction

    def iter_contaminated_sets(self):
        training_dir = os.path.join(self.scenario_path, "training")
        attack_dir = os.path.join(self.scenario_path, "test", "normal_and_attack")
        attack_files = os.listdir(attack_dir)
        attack_files.sort()
        attack_mixins, test_split_files = split_list(attack_files, self.max_attack_fraction)
        for i, num_attacks in enumerate(range(0, len(attack_files), self.step_size)):
            workdir = os.path.join(self.tmp_dir, uuid.uuid4().__str__())
            os.mkdir(workdir)
            attacks = attack_files[:num_attacks]
            copytree(os.path.join(self.scenario_path, "training"), os.path.join(workdir, "training"))
            copytree(os.path.join(self.scenario_path, "validation"), os.path.join(workdir, "validation"))
            os.mkdir(os.path.join(workdir, "test"))
            copytree(os.path.join(self.scenario_path, "test", "normal"), os.path.join(workdir, "test", "normal"))
            os.mkdir(os.path.join(workdir, "test", "normal_and_attack"))
            for f in test_split_files:
                copyfile(os.path.join(attack_dir, f), os.path.join(workdir, "test", "normal_and_attack", f))
            for f in attacks:
                copyfile(os.path.join(attack_dir, f), os.path.join(workdir, "training", f))
            yield {
                "path": workdir,
                "num_attacks": num_attacks,
                "attacks": attacks
            }


DECISION_ENGINES = {de.__name__: de for de in [AE, Stide, Som]}


class Experiment:
    def __init__(self, parameters, mlflow: MlflowClient):
        self.tmp_dir = "tmp"
        self._tmp_results_df = None
        self.parameters = parameters
        self.mlflow = mlflow
        self.scenarios = self._get_param("scenarios", exp_type=list)

    def start(self, start_at=0, dry_run=False):
        i = -1
        for scenario_name in self.scenarios:
            scenario_path = f"{LID_DS_BASE_PATH}/{LID_DS_VERSION}/{scenario_name}"
            attack_mixin = AttackMixin(tmp_dir=self.tmp_dir, scenario_path=scenario_path,
                                            **self.parameters["attack_mixin"])
            for contamined_set in attack_mixin.iter_contaminated_sets():
                i=i+1
                if i < start_at:
                    continue
                if dry_run:
                    print("Dry Run: ", contamined_set)
                    continue
                with mlflow.start_run() as run:
                    mlflow.log_params(convert_mlflow_dict(contamined_set))
                    mlflow.log_params(convert_mlflow_dict(self.parameters))
                    mlflow.log_params(convert_mlflow_dict({"iteration":i}))
                    additional_params, results, ids = self.train_test(contamined_set["path"])
                    self.log_artifacts(ids)
                    mlflow.log_params(convert_mlflow_dict(additional_params))
                    for metric_key, value in convert_mlflow_dict(results).items():
                        try:
                            value = float(value)
                        except ValueError:
                            print("Skip metric", metric_key)
                            continue
                        mlflow.log_metric(metric_key, value)

    def _get_param(self, *keys, default=None, required=True, exp_type=None):
        if default is not None:
            required = False
        cur_obj = self.parameters
        cur_key = ""
        for key in keys:
            if not isinstance(cur_obj, dict):
                raise ValueError("Parameter '%s' is not a dict." % cur_key)
            if key not in cur_obj:
                if not required:
                    return default
                else:
                    raise ValueError("Cannot find parameter for key %s at %s" % (key, cur_key))
            cur_obj = cur_obj[key]
            cur_key = "%s.%s" % (cur_key, key)
        if exp_type is not None and not isinstance(cur_obj, exp_type):
            raise ValueError("Parameter %s is not of expected type %s" % (cur_key, exp_type))
        return cur_obj

    def train_test(self, scenario_path):
        # just load < closing system calls for this example
        dataloader = dataloader_factory(scenario_path, direction=Direction.BOTH)
        DecisionEngineClass = DECISION_ENGINES[self.parameters["decision_engine"]["name"]]

        syscall_name = SyscallName()
        feature_name = self._get_param("features", "name")
        if feature_name == "OneHotEncodingNgram":
            embedding = OneHotEncoding(syscall_name)
        elif feature_name == "IntEmbeddingNgram":
            embedding = IntEmbedding(syscall_name)
        elif feature_name == "W2VEmbedding":
            embedding = W2VEmbedding(syscall_name,
                                     window_size=self._get_param("features", "window_size"),
                                     vector_size=self._get_param("features", "vector_size"),
                                     epochs = self._get_param("features", "epochs")
                                     )
        else:
            raise ValueError("%s is not a valid name" % feature_name)
        thread_aware = self._get_param("features", "thread_aware", default=True)
        n_gram_length = self._get_param("features", "n_gram_length", default=7)
        ngram = Ngram([embedding], thread_aware, n_gram_length)

        decision_engine_args = self._get_param("decision_engine", "args", default={}, exp_type=dict)
        decision_engine = DecisionEngineClass(ngram, **decision_engine_args)
        if DecisionEngineClass == DECISION_ENGINES["Stide"]:
            decision_engine = StreamSum(decision_engine, False, 500, False)
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
            "config": ids.get_config_tree_links(),
            "scenario": scenario_path,
            'dataset': "2021",
            "direction": dataloader.get_direction_string(),
        }
        return additional_parameters, results, ids

    def calc_extended_results(self, performance: Performance):
        results = performance.get_results()
        cm = ConfusionMatrix(tn=results["true_negatives"], fp=results["false_positives"], tp=results["true_positives"],
                             fn=results["false_negatives"])
        metrics = cm.calc_unweighted_measurements()
        return {**results, **metrics}

    def log_artifacts(self, ids):
        outfile = tempfile.mktemp()
        with open(outfile, "wb") as f:
            pickle.dump(ids, f)
        mlflow.log_artifact(outfile, "ids.pickle")
        #outfile = tempfile.mktemp()
        #with open(outfile, "w") as f:
        #    yaml.dump(self.parameters, f)
        #mlflow.log_artifact(outfile, "config.yaml")
        mlflow.log_dict(self.parameters, "config.json")

    @staticmethod
    def continue_run(mlflow_client: MlflowClient, run_id: str, dry_run=False):
        run = mlflow_client.get_run(run_id)
        artifact_uri = run.info.artifact_uri
        config_json = mlflow.artifacts.load_dict(artifact_uri + "/config.json")
        iteration = int(run.data.params.get("iteration"))
        Experiment(config_json, mlflow_client).start(iteration+1, dry_run=dry_run)


def split_list(l, fraction_sublist1: float):
    if fraction_sublist1 < 0 or fraction_sublist1 > 1:
        raise ValueError("Argument fraction_sublist1 must be between 0 and 1, but is: %s" % fraction_sublist1)
    size = len(l)
    split_at = math.floor(fraction_sublist1 * size)
    return l[:split_at], l[split_at:]

def convert_mlflow_dict(nested_dict: dict):
    mlflow_dict = {}
    for key, value  in nested_dict.items():
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


def main():
    mlflow_client = MlflowClient()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=False, help="Experiment config yaml file.")
    parser.add_argument("--start-at", "-s", required=False, help="Start at iteration", type=int, default=0)
    parser.add_argument("--continue-run", "-r", required=False, help="Continue run id from mlflow", type=str)
    parser.add_argument("--dry-run", action="store_true", default=False, help="If set, only attack mixin datasets will be created, and no Anomaly Detection is done. Useful for debugging.")
    args = parser.parse_args()
    if args.config is None and args.continue_run is None:
        print("Either --config or --continue-run must be set")
        parser.print_help()
        exit(1)
    if args.config is not None and args.continue_run is not None:
        print("Only one of --config and --continue-run can be set")
        parser.print_help()
        exit(1)
    if args.config is not None:
        with open(args.config) as f:
            exp_parameters = yaml.safe_load(f)
        start_at = args.start_at
        Experiment(exp_parameters, mlflow=mlflow_client).start(start_at, dry_run=args.dry_run)
    if args.continue_run is not None:
        Experiment.continue_run(mlflow_client, args.continue_run, args.dry_run)

