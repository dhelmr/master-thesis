import abc
import typing

import numpy as np
import pandas
from numpy import e

from algorithms.building_block import BuildingBlock, IDSPhase
from dataloader.syscall import Syscall
from tsa.histogram import Histogram


class AnalyserBB(BuildingBlock):
    def __init__(self, input_bb: BuildingBlock, update_interval=1000, fixed_stops=None, test_phase: bool = False):
        super().__init__()
        if fixed_stops is None:
            fixed_stops = set()
        self._input = input_bb
        self._dependency_list = [self._input]
        self._update_interval = update_interval
        self._current_i = 0
        self._stats = []
        self._fixed_stops = set(fixed_stops)
        self._test_phase = test_phase

    def _calculate(self, syscall: Syscall):
        inp = self._input.get_result(syscall)
        if self._test_phase and self._ids_phase == IDSPhase.TEST:
            self._add_test_input(syscall, inp)
        return inp

    @abc.abstractmethod
    def _add_input(self, syscall, inp):
        raise NotImplementedError()

    def _add_test_input(self, syscall, inp):
        pass

    def train_on(self, syscall: Syscall):
        inp = self._input.get_result(syscall)
        self._add_input(syscall, inp)
        self._current_i += 1
        if self._current_i in self._fixed_stops or \
                (self._update_interval is not None and self._current_i % self._update_interval == 0):
            self.__update_stats()

    def fit(self):
        self.__update_stats(is_last=True)

    def __update_stats(self, is_last=False):
        cur_stats = self._make_stats()
        if isinstance(cur_stats, dict):
            self.__add_stats(cur_stats, is_last)
        if isinstance(cur_stats, list):
            for s in cur_stats:
                self.__add_stats(s, is_last)

    def __add_stats(self, stats_dict, is_last: bool):
        stats_dict["syscalls"] = self._current_i
        stats_dict["is_last"] = is_last
        self._stats.append(stats_dict)

    def depends_on(self) -> list:
        return self._dependency_list

    @abc.abstractmethod
    def _make_stats(self) -> typing.Union[typing.List[dict], dict]:
        raise NotImplementedError()

    def get_stats(self):
        df = pandas.DataFrame(self._stats)
        return df

    @property
    def test_phase(self):
        return self._test_phase


class TrainingSetAnalyser(AnalyserBB):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._histogram = Histogram()

    def _add_input(self, inp):
        if inp is None:
            # TODO
            return
        self._histogram.add(inp)

    def _make_stats(self):
        uniq = self._histogram.unique_elements()
        return {
            # "variance": statistics.pvariance(self._data_list),
            # "stdev": statistics.pstdev(self._data_list),
            "unique": uniq,
            "total": len(self._histogram),
            "u/t": uniq / len(self._histogram),
            "entropy": self._histogram.entropy(base=e),
            "simpson_index": self._histogram.simpson_index()
        }


def entropy(data, base=None):
    value, counts = np.unique(np.array(data), return_counts=True, axis=0)
    print(value, counts)
    norm_counts = counts / counts.sum()
    base = e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()
