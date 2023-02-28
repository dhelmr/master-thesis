import math
import os
import csv
import json
import random

import pandas as pd
from tqdm import tqdm

from dataloader.direction import Direction
from dataloader.recording_2019 import Recording2019
from dataloader.base_data_loader import BaseDataLoader


class DataLoader2019(BaseDataLoader):

    def __init__(self, scenario_path: str, num_attacks: int, direction: Direction = Direction.OPEN,
                 validation_ratio: float = 0.2, cont_ratio: float = 0.2, shuffle_cont_seed=0,
                 training_size=200, validation_size=50):
        super().__init__(scenario_path)
        self.scenario_path = scenario_path
        self._runs_path = os.path.join(scenario_path, 'runs.csv')
        self._normal_recordings = None
        self._exploit_recordings = None
        self._contamined_recordings = None
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

        self.extract_recordings()

    def training_data(self) -> list:
        return self._normal_recordings[:self._training_size] + self._contamined_recordings

    def validation_data(self) -> list:
        return self._normal_recordings[self._training_size:self._training_size + self._validation_size]

    def test_data(self) -> list:
        return self._normal_recordings[self._training_size + self._validation_size:] + self._exploit_recordings

    def extract_recordings(self):
        with open(self._runs_path, 'r') as runs_csv:
            recording_reader = csv.reader(runs_csv, skipinitialspace=True)
            next(recording_reader)

            normal_recordings = []
            exploit_recordings = []

            for recording_line in recording_reader:
                recording = Recording2019(recording_line, self.scenario_path, self._direction)
                if not recording.metadata()['exploit']:
                    normal_recordings.append(recording)
                else:
                    exploit_recordings.append(recording)

        self._normal_recordings = normal_recordings
        self._contamined_recordings, self._exploit_recordings = split_list(exploit_recordings, self._cont_ratio)
        random.Random(self._shuffle_cont_seed).shuffle(self._contamined_recordings)
        self._contamined_recordings = self._contamined_recordings[:self.num_attacks]

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


def split_list(l, fraction_sublist1: float):
    if fraction_sublist1 < 0 or fraction_sublist1 > 1:
        raise ValueError("Argument fraction_sublist1 must be between 0 and 1, but is: %s" % fraction_sublist1)
    size = len(l)
    split_at = math.floor(fraction_sublist1 * size)
    return l[:split_at], l[split_at:]
