import csv
import json
import os

from tqdm import tqdm

from dataloader.direction import Direction
from dataloader.recording_2019 import Recording2019, RecordingDataParts
from tsa.dataloaders.combination_dl import yield_successively
from tsa.dataloaders.dataloader_2021 import get_scenario_name
from tsa.dataloaders.tsa_base_dl import TsaBaseDataloader
from tsa.utils import split_list, random_permutation

TRAIN_OFFSET = 200
VAL_OFFSET = 50
DEFAULT_VAL_RATIO = 0.2

class ContaminatedDataLoader2019(TsaBaseDataloader):

    def __init__(self, scenario_path: str, num_attacks: int, direction: Direction = Direction.OPEN,
                 cont_ratio: float = 0.2, permutation_i=0,
                 training_size=None, validation_size=None, test_size=None, true_metadata=False, no_test_attacks: bool = False):
        super().__init__(scenario_path)
        self.scenario_path = scenario_path
        self._runs_path = os.path.join(scenario_path, 'runs.csv')
        self._normal_recordings = None
        self._exploit_recordings = None
        self._contaminated_recordings = None
        self._distinct_syscalls = None
        self._training_size = training_size
        self._validation_size = validation_size
        self._test_size = test_size
        self._direction = direction
        self._cont_ratio = cont_ratio
        self._permutation_i = permutation_i
        self._num_attacks = num_attacks
        self._true_metadata = true_metadata
        self._no_test_attacks = no_test_attacks
        self._initialized = False
        self._metrics = {}

    def training_data(self) -> list:
        self._init_once()
        recordings = list(yield_successively([
            self._normal_recordings[:TRAIN_OFFSET],
            self._contaminated_recordings
        ], limit=self._training_size))
        self._metrics["train_recordings"] = len(recordings)
        return recordings



    def validation_data(self) -> list:
        self._init_once()
        recordings=self._normal_recordings[TRAIN_OFFSET:TRAIN_OFFSET + VAL_OFFSET]
        if self._validation_size is not None:
            recordings = recordings[:self._validation_size]
        self._metrics["validation_recordings"] = len(recordings)
        return recordings

    def test_data(self) -> list:
        self._init_once()
        normal_test_data = self._normal_recordings[TRAIN_OFFSET + VAL_OFFSET:]
        recordings = list(yield_successively([normal_test_data, self._exploit_recordings], limit=self._test_size))
        self._metrics["test_recordings"] = len(recordings)
        return recordings

    def cfg_dict(self):
        return {
            "scenario": get_scenario_name(self.scenario_path),
            "training_size": self._training_size,
            "validation_size": self._validation_size,
            "direction": self._direction,
            "cont_ratio": self._cont_ratio,
            "permutation_i": self._permutation_i,
            "num_attacks": self._num_attacks
        }

    def _extract_recordings(self):
        exploit_recording_lines = []
        with open(self._runs_path, 'r') as runs_csv:
            recording_reader = csv.reader(runs_csv, skipinitialspace=True)
            next(recording_reader)

            normal_recordings = []
            for recording_line in recording_reader:
                recording = Recording2019(recording_line, self.scenario_path, self._direction)
                if not recording.metadata()['exploit']:
                    normal_recordings.append(recording)
                else:
                    exploit_recording_lines.append((recording_line, recording))

        self._normal_recordings = normal_recordings
        training_exploit_lines, test_exploit_lines = split_list(exploit_recording_lines, self._cont_ratio)
        self._exploit_recordings = [recording for (_, recording) in test_exploit_lines]
        if self._no_test_attacks:
            self._exploit_recordings = []

        # create contaminated recordings that will be added to the training phase
        training_exploit_lines = random_permutation(training_exploit_lines, self._num_attacks, self._permutation_i)
        self._contaminated_recordings = []
        for recording_line, _ in training_exploit_lines:
            if not self._true_metadata:
                recording_line[RecordingDataParts.IS_EXECUTING_EXPLOIT] = "false"
            self._contaminated_recordings.append(Recording2019(recording_line, self.scenario_path, self._direction))
        print(self._num_attacks, training_exploit_lines)

    def _init_once(self):
        if self._initialized:
            return
        self._extract_recordings()
        self._initialized = True

    def distinct_syscalls_training_data(self) -> int:
        json_path = 'distinct_syscalls.json'
        try:
            with open(self.scenario_path + json_path, 'r') as distinct_syscalls:
                distinct_json = json.load(distinct_syscalls)
                self._distinct_syscalls = distinct_json['distinct_syscalls']
        except Exception:
            print('Could not load distinct syscalls. Calculating now...')

        if self._distinct_syscalls is not None:
            return self._distinct_syscalls
        else:
            syscall_dict = {}
            description = 'Calculating distinct syscalls'.rjust(25)
            for recording in tqdm(self.training_data(), description, unit=' recording'):
                for syscall in recording.syscalls():
                    if syscall.name() in syscall_dict:
                        continue
                    else:
                        syscall_dict[syscall.name()] = True
            self._distinct_syscalls = len(syscall_dict)
            with open(self.scenario_path + json_path, 'w') as distinct_syscalls:
                json.dump({'distinct_syscalls': self._distinct_syscalls}, distinct_syscalls)
            return self._distinct_syscalls

    def metrics(self):
        return self._metrics

    def artifact_dict(self):
        return {
            "attack_names": [r.name for r in self._contaminated_recordings],
        }

    def get_val_ratio(self):
        if self._training_size is None or self._validation_size is None:
            return DEFAULT_VAL_RATIO
        return self._validation_size / (self._validation_size + self._training_size)
