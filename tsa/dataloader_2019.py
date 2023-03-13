import os
import csv
import json
import random

from tqdm import tqdm

from dataloader.direction import Direction
from dataloader.recording_2019 import Recording2019, RecordingDataParts
from dataloader.base_data_loader import BaseDataLoader
from tsa.dataloader_2021 import get_scenario_name
from tsa.utils import split_list


class ContaminatedDataLoader2019(BaseDataLoader):

    def __init__(self, scenario_path: str, num_attacks: int, direction: Direction = Direction.OPEN,
                 validation_ratio: float = 0.2, cont_ratio: float = 0.2, shuffle_cont_seed=0,
                 training_size=200, validation_size=50):
        super().__init__(scenario_path)
        self.scenario_path = scenario_path
        self._runs_path = os.path.join(scenario_path, 'runs.csv')
        self._normal_recordings = None
        self._exploit_recordings = None
        self._contaminated_recordings = None
        self._distinct_syscalls = None
        self._training_size = training_size
        self._validation_size = validation_size
        self._direction = direction
        self._cont_ratio = cont_ratio
        self._shuffle_cont_seed = shuffle_cont_seed
        self._validation_ratio = validation_ratio
        self._num_attacks = num_attacks
        if validation_ratio < 0 or validation_ratio > 1:
            raise ValueError("validation_ratio must be in interval [0,1]")
        self._initialized = False

    def training_data(self) -> list:
        self._init_once()
        return self._normal_recordings[:self._training_size] + self._contaminated_recordings

    def validation_data(self) -> list:
        self._init_once()
        return self._normal_recordings[self._training_size:self._training_size + self._validation_size]

    def test_data(self) -> list:
        self._init_once()
        return self._normal_recordings[self._training_size + self._validation_size:] + self._exploit_recordings

    def cfg_dict(self):
        return {
            "scenario": get_scenario_name(self.scenario_path),
            "training_size": self._training_size,
            "validation_size": self._validation_size,
            "direction": self._direction,
            "cont_ratio": self._cont_ratio,
            "shuffle_cont_seed": self._shuffle_cont_seed,
            "validation_ratio": self._validation_ratio,
            "num_attacks": self._num_attacks,
            "attack_names": [r.name for r in self._contaminated_recordings]
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

        # create contaminated recordings that will be added to the training phase
        random.Random(self._shuffle_cont_seed).shuffle(training_exploit_lines)
        # TODO handle case num_attacks > len(training_exploit_lines)
        training_exploit_lines = training_exploit_lines[:self._num_attacks]
        self._contaminated_recordings = []
        for recording_line, _ in training_exploit_lines:
            recording_line[RecordingDataParts.IS_EXECUTING_EXPLOIT] = "false"
            self._contaminated_recordings.append(Recording2019(recording_line, self.scenario_path, self._direction))



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


