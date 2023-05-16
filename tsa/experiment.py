import dataclasses
import itertools
from typing import List

import mlflow
import os
from mlflow import MlflowClient

from algorithms.ids import IDS
from algorithms.performance_measurement import Performance
from dataloader.base_data_loader import BaseDataLoader
from dataloader.direction import Direction
from tsa.building_block_builder import IDSPipelineBuilder
from tsa.confusion_matrix import ConfusionMatrix
from tsa.dataloader_2019 import ContaminatedDataLoader2019
from tsa.dataloader_2021 import ContaminatedDataLoader2021
from tsa.utils import access_cfg

try:
    LID_DS_BASE_PATH = os.environ['LID_DS_BASE']
except KeyError as exc:
    raise ValueError("No LID-DS Base Path given."
                     "Please specify as argument or set Environment Variable "
                     "$LID_DS_BASE") from exc


@dataclasses.dataclass
class RunConfig:
    lid_ds_version: str
    scenario: str
    num_attacks: int
    iteration: int
    permutation_i: int

    def to_dict(self):
        return dataclasses.asdict(self)


class Experiment:
    def __init__(self, parameters, mlflow: MlflowClient, name):
        self._tmp_results_df = None
        self.parameters = parameters
        self.mlflow = mlflow
        self.scenarios = self._get_param("scenarios", exp_type=list)
        self.name = name

    def run_configurations(self) -> List[RunConfig]:
        configs = []
        num_attacks_range = self._get_param("attack_mixin", "num_attacks", required=False)
        if num_attacks_range is None:
            max_attacks = self._get_param("attack_mixin", "max_attacks", exp_type=int)
            num_attacks_range = range(max_attacks + 1)
        permutation_i_values = self._get_param("attack_mixin", "permutation_i", required=True)
        if not isinstance(permutation_i_values, list):
            if not str(permutation_i_values).isdigit():
                raise ValueError("Invalid value for permutation_i: %s" % permutation_i_values)
            permutation_i_values = [permutation_i_values]
        iteration = 0
        for scenario in self.scenarios:
            lid_ds_version, scenario_name = scenario.split("/")

            for num_attacks, permutation_i in itertools.product(num_attacks_range, permutation_i_values):
                cfg = RunConfig(
                    num_attacks=num_attacks,
                    iteration=iteration,
                    scenario=scenario_name,
                    lid_ds_version=lid_ds_version,
                    permutation_i=permutation_i
                )
                configs.append(cfg)
                iteration += 1
        return configs

    def start(self, start_at=0, dry_run=False, num_runs=None):
        mlflow.set_experiment(self.name)
        dataloader_config = self._get_dataloader_cfg()
        i = -1
        current_run = 0
        for run_cfg in self.run_configurations():
            print("Execute run", run_cfg.to_dict())
            dataloader_class = self._get_dataloader_cls(run_cfg.lid_ds_version)
            scenario_path = f"{LID_DS_BASE_PATH}/{run_cfg.lid_ds_version}/{run_cfg.scenario}"

            dataloader = dataloader_class(scenario_path, num_attacks=run_cfg.num_attacks, direction=Direction.BOTH,
                                          permutation_i=run_cfg.permutation_i, **dataloader_config)
            i = i + 1
            if i < start_at:
                continue
            if dry_run:
                print(i, "Dry Run: ", dataloader.__dict__)
                continue
            current_run += 1
            if num_runs is not None and current_run > num_runs:
                print("Reached total number of runs (%s)" % num_runs)
                return
            with mlflow.start_run() as run:
                mlflow.log_params(convert_mlflow_dict(run_cfg.to_dict()))
                mlflow.log_params(convert_mlflow_dict(dataloader.cfg_dict(), "dataloader"))
                mlflow.log_dict(self.parameters, "config.json")
                additional_params, results, ids = self.train_test(dataloader, run_cfg)
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

    def _get_dataloader_cfg(self):
        return self._get_param("attack_mixin", "dataloader", exp_type=dict)

    def train_test(self, dataloader: BaseDataLoader, run_cfg: RunConfig):
        builder = IDSPipelineBuilder()  # TODO change experiment yaml format; add own key for pipeline
        building_block_configs = self._get_param("ids", exp_type=list)
        decider = builder.build_all(building_block_configs)

        ids = IDS(data_loader=dataloader,
                  resulting_building_block=decider,
                  create_alarms=True,
                  plot_switch=False)

        print("at evaluation:")

        performance = ids.detect()

        results = self.calc_extended_results(performance)
        additional_parameters = {
            f"tsa{i + 1}": analyser.get_analyse_result() for i, analyser in enumerate(builder.analysers)
        }
        # additional_parameters["config"] = ids.get_config_tree_links(),
        return additional_parameters, results, ids

    def calc_extended_results(self, performance: Performance):
        results = performance.get_results()
        cm = ConfusionMatrix(tn=results["true_negatives"], fp=results["false_positives"], tp=results["true_positives"],
                             fn=results["false_negatives"])
        metrics = cm.calc_unweighted_measurements()
        return {"ids": results, "cm": metrics}

    def _get_param(self, *args, **kwargs):
        return access_cfg(self.parameters, *args, **kwargs)


def convert_mlflow_dict(nested_dict: dict, prefix=None):
    mlflow_dict = {}
    for key, value in nested_dict.items():
        if prefix is not None:
            key = f"{prefix}.{key}"
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
