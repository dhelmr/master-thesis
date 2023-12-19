from algorithms.building_block import BuildingBlock, IDSPhase
from algorithms.ids import IDS
from dataloader.syscall import Syscall
from tsa.dataloaders.tsa_base_dl import TsaBaseDataloader
from tsa.diagnosis.preprocessing import OutlierDetector
from tsa.experiment import Experiment


class UnsupervisedEvaluator(BuildingBlock):
    def __init__(self, building_block: OutlierDetector):
        super().__init__()
        if not isinstance(building_block, OutlierDetector):
            raise ValueError("Input building block must be of class OutlierDetector")
        self._input = building_block
        self._dependency_list = [building_block]
        self._predictions = []
        building_block.anomaly_return_value = True

    def _calculate(self, syscall: Syscall):
        if self._ids_phase == IDSPhase.TEST:
            self.set_ids_phase(IDSPhase.TRAINING)
        outlier_result = self._input.get_result(syscall)
        if outlier_result == True:
            return True
        return False

    def depends_on(self) -> list:
        return self._dependency_list

    def is_decider(self):
        return True


class UnsupervisedDataLoader(TsaBaseDataloader):
    def __init__(self, wrapped_dataloader: TsaBaseDataloader):
        super().__init__(wrapped_dataloader.scenario_path)
        self.dl = wrapped_dataloader

    def training_data(self) -> list:
        return self.dl.training_data()

    def validation_data(self) -> list:
        return self.dl.validation_data()

    def test_data(self) -> list:
        # return training data again because only the training set should be evaluated
        return self.dl.training_data()

    def extract_recordings(self, category: str) -> list:
        return self.dl.extract_recordings(category)

    def collect_metadata(self) -> dict:
        return self.dl.collect_metadata()

    def get_direction_string(self):
        return self.dl.get_direction_string()

    def cfg_dict(self):
        # TODO implement own superclass for dataloaders
        return self.dl.cfg_dict()

    def metrics(self):
        return self.dl.metrics()

    def artifact_dict(self):
        return self.dl.artifact_dict()


class UnsupervisedExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_test(self, dataloader, run_config, builder):
        ids_cfg = self._get_param("ids", exp_type=list)
        last_bb = builder.build_all(ids_cfg)
        if not isinstance(last_bb, OutlierDetector):
            raise ValueError(
                "Expect the resulting block of the experiment to be an outlier detector."
            )
        evaluator = UnsupervisedEvaluator(last_bb)
        only_training_dl = UnsupervisedDataLoader(dataloader)
        ids = IDS(
            data_loader=only_training_dl,
            resulting_building_block=evaluator,
            create_alarms=True,
            plot_switch=False,
        )
        performance = ids.detect()
        results = self.calc_extended_results(performance)
        return {}, results, ids

    def _get_dataloader_cfg(self):
        cfg = super()._get_dataloader_cfg()
        cfg["true_metadata"] = True
        return cfg
