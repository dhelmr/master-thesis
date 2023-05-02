import mlflow
import os
from mlflow import MlflowClient

from algorithms.ids import IDS
from algorithms.performance_measurement import Performance
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

    def train_test(self, dataloader):
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
        additional_parameters["config"] = ids.get_config_tree_links(),
        return additional_parameters, results, ids

    def calc_extended_results(self, performance: Performance):
        results = performance.get_results()
        cm = ConfusionMatrix(tn=results["true_negatives"], fp=results["false_positives"], tp=results["true_positives"],
                             fn=results["false_negatives"])
        metrics = cm.calc_unweighted_measurements()
        return {"ids": results, "cm": metrics}

    def _get_param(self, *args, **kwargs):
        return access_cfg(self.parameters, *args, **kwargs)

    def log_artifacts(self, ids):
        mlflow.log_dict(self.parameters, "config.json")

    @staticmethod
    def continue_run(mlflow_client: MlflowClient, run_id: str, dry_run=False):
        run = mlflow_client.get_run(run_id)
        artifact_uri = run.info.artifact_uri
        config_json = mlflow.artifacts.load_dict(artifact_uri + "/config.json")
        iteration = int(run.data.params.get("iteration"))
        Experiment(config_json, mlflow_client).start(iteration + 1, dry_run=dry_run)


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
