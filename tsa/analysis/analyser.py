import abc

import numpy as np
import pandas
from numpy import e
from pandas import DataFrame

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall

from tsa.unsupervised.mixed_model import Histogram


class AnalyserBB(BuildingBlock):
    def __init__(self, input_bb: BuildingBlock, update_interval = 1000):
        super().__init__()
        self._input = input_bb
        self._dependency_list = [self._input]
        self._update_interval = update_interval
        self._current_i = 0
        self._stats = []

    def _calculate(self, syscall: Syscall):
        return self._input.get_result(syscall)

    @abc.abstractmethod
    def _add_input(self, inp):
        raise NotImplementedError()
    def train_on(self, syscall: Syscall):
        inp = self._input.get_result(syscall)
        self._add_input(inp)
        self._current_i += 1
        if self._current_i % self._update_interval == 0:
            cur_stats = self._make_stats()
            cur_stats["syscalls"] = self._current_i
            self._stats.append(cur_stats)

    def depends_on(self) -> list:
        return self._dependency_list

    @abc.abstractmethod
    def _make_stats(self):
        raise NotImplementedError()

    def get_stats(self):
        df = pandas.DataFrame(self._stats)
        return df


class TrainingSetAnalyser(AnalyserBB):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._histogram = Histogram()
        self._data_list = []

    def _add_input(self, inp):
        if inp is None:
            # TODO
            return
        self._histogram.add(inp)
        self._data_list.append(inp)
    def _make_stats(self):
        uniq = self._histogram.unique_elements()
        return {
            # "variance": statistics.pvariance(self._data_list),
            # "stdev": statistics.pstdev(self._data_list),
            "unique": uniq,
            "total": len(self._histogram),
            "u/t": uniq/len(self._histogram),
            "entropy": self._histogram.entropy(base=e)
        }



def entropy(data, base=None):
    value, counts = np.unique(np.array(data), return_counts=True, axis=0)
    print(value, counts)
    norm_counts = counts / counts.sum()
    base = e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()
