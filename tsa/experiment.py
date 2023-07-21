import copy
import dataclasses
import hashlib
import itertools
import pickle
import pprint
from typing import List, Union, Dict

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
from tsa.dataloaders.combination_dl import CombinationDL
from tsa.dataloaders.filter_dl import FilterDataloader
from tsa.dataloaders.tsa_base_dl import TsaBaseDataloader
from tsa.utils import access_cfg



ScenarioName = str
@dataclasses.dataclass
class CombinedScenario:
    scenarios: Dict[ScenarioName, dict]

@dataclasses.dataclass
class RunConfig:
    parameter_cfg_id: str
    # lid_ds_version: str
    scenario: Union[CombinedScenario, ScenarioName]
    num_attacks: int
    iteration: int
    permutation_i: int

    def to_dict(self):
        d = dataclasses.asdict(self)
        if isinstance(self.scenario, CombinedScenario):
            d["scenario"] = [s for s, _ in self.scenario.scenarios.items()]
        return d


class Experiment:
    def __init__(self, parameter_cfg, mlflow: MlflowClient, mlflow_name):
        self._tmp_results_df = None
        self.parameter_cfg = parameter_cfg
        self.mlflow = mlflow
        self.scenarios = self._get_param("scenarios", exp_type=list)
        self.mlflow_name = mlflow_name
        self.parameter_cfg_id = self._get_param("id", exp_type=str)

    def run_configurations(self) -> List[RunConfig]:
        configs = []
        num_attacks_range = self._get_param("dataloader", "num_attacks", required=False)
        if num_attacks_range is None:
            max_attacks = self._get_param("dataloader", "max_attacks", exp_type=int)
            num_attacks_range = range(max_attacks + 1)
        permutation_i_values = self._get_param("dataloader", "permutation_i", required=True)
        if not isinstance(permutation_i_values, list):
            if not str(permutation_i_values).isdigit():
                raise ValueError("Invalid value for permutation_i: %s" % permutation_i_values)
            permutation_i_values = [permutation_i_values]
        iteration = 0
        for permutation_i, scenario, num_attacks in itertools.product(permutation_i_values,self.scenarios, num_attacks_range):
#            lid_ds_version, scenario_name = scenario.split("/")
            if isinstance(scenario, dict):
                scenario = CombinedScenario(scenarios=scenario)
            cfg = RunConfig(
                parameter_cfg_id=self.parameter_cfg_id,
                num_attacks=num_attacks,
                iteration=iteration,
                scenario=scenario,
                permutation_i=permutation_i
            )
            configs.append(cfg)
            iteration += 1
        return configs

    def start(self, start_at=0, dry_run=False, num_runs=None):
        mlflow.set_experiment(self.mlflow_name)
        i = -1
        current_run = 0
        for run_cfg in self.run_configurations():
            i = i + 1
            if i < start_at:
                continue
            dataloader = self._get_dataloader_cls(run_cfg)
            if dry_run:
                print(i, "Dry Run: ", dataloader.__dict__)
                continue
            print("Execute run", run_cfg.to_dict())
            current_run += 1
            if num_runs is not None and current_run > num_runs:
                print("Reached total number of runs (%s)" % num_runs)
                return
            with mlflow.start_run() as run:
                mlflow.log_params(convert_mlflow_dict(run_cfg.to_dict()))
                mlflow.log_params(convert_mlflow_dict(dataloader.cfg_dict(), "dataloader"))
                print(self.parameter_cfg)
                mlflow.log_dict(self.parameter_cfg, "config.json")
                self._log_ids_cfg()
                dataloader_context = copy.deepcopy(dataloader.cfg_dict())
                if dataloader_context["num_attacks"] == 0:
                    # the cache context should be the same for all runs with num_attacks=0, because permutation_i has no effect
                    del dataloader_context["permutation_i"]
                builder = IDSPipelineBuilder(cache_context=str(dataloader_context))
                additional_params, results, ids = self.train_test(dataloader, run_cfg, builder)
                pprint.pprint(results)
                # update dataloader reference because the ids might have been loaded from cache
                dataloader: TsaBaseDataloader = ids._data_loader
                mlflow.log_params(convert_mlflow_dict(additional_params))
                self._log_metrics(dataloader.metrics(), "dataloader")
                dl_artifacts = dataloader.artifact_dict()
                if dl_artifacts != {}:
                    mlflow.log_dict(dl_artifacts, "dataloader_artifacts.json")
                self._log_metrics(results)

    def _log_metrics(self, metrics_dict, prefix=None):
        for metric_key, value in convert_mlflow_dict(metrics_dict, prefix).items():
            try:
                value = float(value)
            except ValueError:
                print("Skip metric", metric_key)
                continue
            mlflow.log_metric(metric_key, value)

    def _make_dl_from_scenario(self, scenario: ScenarioName, cfg: Dict):
        try:
            lid_ds_base = os.environ['LID_DS_BASE']
        except KeyError as exc:
            raise ValueError("No LID-DS Base Path given."
                             "Please specify as argument or set Environment Variable "
                             "$LID_DS_BASE") from exc
        lid_ds_version = scenario.split("/")[0]
        if lid_ds_version == "LID-DS-2019":
            cls = ContaminatedDataLoader2019
        elif lid_ds_version == "LID-DS-2021":
            cls = ContaminatedDataLoader2021
        else:
            raise ValueError("%s is not supported." % lid_ds_version)
        scenario_path = f"{lid_ds_base}/{scenario}"
        dataloader = cls(scenario_path, **cfg)
        return dataloader

    def _get_dataloader_cls(self, run_cfg: RunConfig):
        base_cfg = self._get_dataloader_cfg()
        base_cfg = {
            **base_cfg,
            "permutation_i": run_cfg.permutation_i,
            "num_attacks": run_cfg.num_attacks
        }
        if isinstance(run_cfg.scenario, ScenarioName):
            dataloader = self._make_dl_from_scenario(run_cfg.scenario, base_cfg)
        elif isinstance(run_cfg.scenario, CombinedScenario):
            dls = []
            for i, scenario_cfg in enumerate(run_cfg.scenario.scenarios.items()):
                scenario_name, scenario_dl_cfg = scenario_cfg
                dataloader_cfg = copy.deepcopy(base_cfg)
                dataloader_cfg = {
                    **scenario_dl_cfg,
                    **dataloader_cfg
                }
                if i == 0: # first dataloader has mum_attacks from run cfg
                    part_dataloader = self._make_dl_from_scenario(scenario_name, dataloader_cfg)
                else:
                    dataloader_cfg["num_attacks"] = 0
                    part_dataloader = self._make_dl_from_scenario(scenario_name, dataloader_cfg)
                dls.append(part_dataloader)
            dataloader = CombinationDL(dls)
        else:
            raise ValueError("Not supported scenario specification: %s" % run_cfg.scenario)
        dataloader = FilterDataloader(dataloader, **self._get_param("dataloader", "filter", default={}))
        return dataloader

    def _get_dataloader_cfg(self):
        cfg = self._get_param("dataloader", "base", exp_type=dict)
        if "direction" not in cfg:
            cfg["direction"] = Direction.BOTH # TODO: make configurable
        return cfg

    def train_test(self, dataloader: BaseDataLoader, run_cfg: RunConfig, builder):
        building_block_configs = self._get_param("ids", exp_type=list)
        decider = builder.build_all(building_block_configs)

        use_cache = self._get_param("cache", default=False, exp_type=bool)
        ids = None
        if use_cache:
            ids = self._load_from_cache(dataloader, run_cfg)
        if ids is None:
            ids = IDS(data_loader=dataloader,
                      resulting_building_block=decider,
                      create_alarms=True,
                      plot_switch=False)
        if use_cache:
            self._serialize_ids(ids, dataloader, run_cfg)

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
        return copy.deepcopy(access_cfg(self.parameter_cfg, *args, **kwargs))

    def _log_ids_cfg(self):
        ids_cfg = self._get_param("ids", exp_type=list)
        for bb_cfg in ids_cfg:
            self._log_bb(bb_cfg)

    def _log_bb(self, bb_cfg):
        if "split" in bb_cfg:
            for _, cfgs in bb_cfg["split"].items():
                for cfg in cfgs:
                    self._log_bb(cfg)
        else:
            name = bb_cfg["name"]
            if "mlflow_name" in bb_cfg:
                name = bb_cfg["mlflow_name"]
            if "args" in bb_cfg:
                params = convert_mlflow_dict(bb_cfg["args"], prefix=name)
                mlflow.log_params(params)

    def _serialization_path(self, dataloader, run_cfg):
        if "IDS_CACHE_PATH" not in os.environ:
            raise KeyError("$IDS_CACHE_PATH must be set if caching of ids is used")
        base_path = os.environ["IDS_CACHE_PATH"]
        cfg = self.parameter_cfg
        context = str(cfg) + "||" + str(dataloader.cfg_dict()) + "||" + str(run_cfg.to_dict())
        context_hash = hashlib.md5(context.encode()).hexdigest()
        return os.path.join(base_path, context_hash+".ids.pickle")

    def _serialize_ids(self, ids: IDS, dataloader, run_cfg):
        path = self._serialization_path(dataloader, run_cfg)
        if os.path.exists(path):
            print("IDS Serialization already exists, skip.", path)
            return
        print("Write ids to", path)
        with open(path, "wb") as f:
            pickle.dump(ids, f)

    def _load_from_cache(self, dataloader, run_cfg):
        path = self._serialization_path(dataloader, run_cfg)
        if not os.path.exists(path):
            print("IDS Serialization does not exist, skip.", path)
            return None
        print("Load IDS from", path)
        with open(path, "rb") as f:
            return pickle.load(f)


def convert_mlflow_dict(entry: Union[dict, list], prefix=None):
    mlflow_dict = {}
    if isinstance(entry, dict):
        key_values = [kv for kv in entry.items()]
    elif isinstance(entry, list):
        key_values = [(i, item) for i, item in enumerate(entry)]
    else:
        raise ValueError("Unexpected type: %s" % type(entry))
    for key, value in key_values:
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
