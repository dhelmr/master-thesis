from typing import List

from dataloader.data_loader_2021 import DataLoader2021, RecordingType, TRAINING, TEST
from dataloader.direction import Direction
from dataloader.recording_2021 import Recording2021
from tsa.utils import split_list, random_permutation
import random


def get_scenario_name(scenario_path):
    last = scenario_path.split("/")[-2:]
    return "/".join(last)


class ContaminatedRecording2021(Recording2021):
    def __init__(self, original_recording: Recording2021, true_metadata: bool):
        super().__init__(original_recording.path, original_recording.name, original_recording._direction)
        self._true_metadata = true_metadata

    def metadata(self) -> dict:
        metadata = super().metadata()
        if not self._true_metadata:
            metadata["exploit"] = False
            metadata["exploit_name"] = "no-exploit"
            metadata["time"]["exploit"] = []
        return metadata


class ContaminatedDataLoader2021(DataLoader2021):
    def __init__(self, scenario_path: str, num_attacks: int, direction: Direction = Direction.OPEN,
                 validation_ratio: float = 0.2, cont_ratio: float = 0.2, permutation_i=0,
                 training_size=200, validation_size=50, true_metadata=False):
        super().__init__(scenario_path, direction)
        self._num_attacks = num_attacks
        self._validation_ratio = validation_ratio
        self._cont_ratio = cont_ratio
        self._permutation_i = permutation_i
        self._true_metadata = true_metadata
        self._init_contaminated()

    def _init_contaminated(self):
        test_recordings: List[Recording2021] = super().extract_recordings(
            category=TEST
        )
        exploits = [t for t in test_recordings if t.metadata()["exploit"] == True]
        for_training, _ = split_list(exploits, self._cont_ratio)
        contaminated_recording_names = [r.name for r in for_training]
        # TODO handle case num_attacks > len(contaminated_recording_names)
        self._contaminated_recordings = set(
            random_permutation(contaminated_recording_names, self._num_attacks, self._permutation_i))
        self._exclude_recordings = set(
            [r for r in contaminated_recording_names if r not in self._contaminated_recordings])
        print(self._exclude_recordings, self._contaminated_recordings)

    def training_data(self, recording_type: RecordingType = None) -> list:
        training_data = super().training_data()
        test_data = super().test_data()
        contaminated_data = [ContaminatedRecording2021(r, true_metadata=self._true_metadata) for r in test_data if
                             r.name in self._contaminated_recordings]
        return training_data + contaminated_data

    def test_data(self, recording_type: RecordingType = None) -> list:
        test_data = super().test_data()
        return [r for r in test_data if r.name not in self._exclude_recordings]

    def cfg_dict(self):
        return {
            "scenario": get_scenario_name(self.scenario_path),
            "training_size": -1,
            "validation_size": -1,
            "direction": self._direction,
            "cont_ratio": self._cont_ratio,
            "permutation_i": self._permutation_i,
            "validation_ratio": self._validation_ratio,
            "num_attacks": self._num_attacks,
            "attack_names": list(self._contaminated_recordings)
        }
