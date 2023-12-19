from typing import List

from dataloader.data_loader_2021 import DataLoader2021, RecordingType, TEST
from dataloader.direction import Direction
from dataloader.recording_2021 import Recording2021
from tsa.dataloaders.combination_dl import yield_successively
from tsa.dataloaders.tsa_base_dl import TsaBaseDataloader
from tsa.utils import split_list, random_permutation


def get_scenario_name(scenario_path):
    last = scenario_path.split("/")[-2:]
    return "/".join(last)


class ContaminatedRecording2021(Recording2021):
    def __init__(self, original_recording: Recording2021, true_metadata: bool):
        super().__init__(
            original_recording.path,
            original_recording.name,
            original_recording._direction,
        )
        self._true_metadata = true_metadata

    def metadata(self) -> dict:
        metadata = super().metadata()
        if not self._true_metadata:
            metadata["exploit"] = False
            metadata["exploit_name"] = "no-exploit"
            metadata["time"]["exploit"] = []
        return metadata


class ContaminatedDataLoader2021(DataLoader2021, TsaBaseDataloader):
    def __init__(
        self,
        scenario_path: str,
        num_attacks: int,
        direction: Direction = Direction.OPEN,
        cont_ratio: float = 0.2,
        permutation_i=0,
        training_size=None,
        validation_size=None,
        test_size=None,
        true_metadata=False,
        no_test_attacks: bool = False,
    ):
        super().__init__(scenario_path, direction)
        self._num_attacks = num_attacks
        self._cont_ratio = cont_ratio
        self._training_size = training_size
        self._validation_size = validation_size
        self._test_size = test_size
        self._permutation_i = permutation_i
        self._true_metadata = true_metadata
        self._no_test_attacks = no_test_attacks
        self._metrics = {}
        self._init_contaminated()

    def _init_contaminated(self):
        test_recordings: List[Recording2021] = super().extract_recordings(category=TEST)
        exploits = [t for t in test_recordings if t.metadata()["exploit"] == True]
        for_training, for_test = split_list(exploits, self._cont_ratio)
        contaminated_recording_names = [r.name for r in for_training]
        # TODO handle case num_attacks > len(contaminated_recording_names)
        self._contaminated_recordings = set(
            random_permutation(
                contaminated_recording_names, self._num_attacks, self._permutation_i
            )
        )
        self._exclude_recordings = set(
            [
                r
                for r in contaminated_recording_names
                if r not in self._contaminated_recordings
            ]
        )
        if self._no_test_attacks:
            print("Exclude all")
            self._exclude_recordings = set([r.name for r in for_test])

    def training_data(self, recording_type: RecordingType = None) -> list:
        training_data = super().training_data()
        test_data = super().test_data()
        contaminated_data = [
            ContaminatedRecording2021(r, true_metadata=self._true_metadata)
            for r in test_data
            if r.name in self._contaminated_recordings
        ]

        recordings = list(
            yield_successively(
                [training_data, contaminated_data], limit=self._training_size
            )
        )
        self._metrics["train_recordings"] = len(recordings)
        return recordings

    def test_data(self, recording_type: RecordingType = None) -> list:
        test_data = super().test_data()
        test_recordings = [
            r for r in test_data if r.name not in self._exclude_recordings
        ]
        recordings = test_recordings
        if self._test_size is not None:
            recordings = test_recordings[: self._test_size]
        self._metrics["test_recordings"] = len(recordings)
        return recordings

    def validation_data(self, recording_type: RecordingType = None) -> list:
        val_data = super().validation_data()
        recordings = val_data
        if self._validation_size is not None:
            recordings = val_data[: self._validation_size]
        self._metrics["validation_recordings"] = len(recordings)
        return recordings

    def cfg_dict(self):
        return {
            "scenario": get_scenario_name(self.scenario_path),
            "training_size": -1,
            "validation_size": -1,
            "direction": self._direction,
            "cont_ratio": self._cont_ratio,
            "permutation_i": self._permutation_i,
            "num_attacks": self._num_attacks,
        }

    def metrics(self):
        return self._metrics

    def artifact_dict(self):
        return {
            "attack_names": list(self._contaminated_recordings),
        }
