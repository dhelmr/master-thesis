import numpy as np

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
from tsa.unsupervised.mixed_model import Histogram
from math import e


class TrainingSetAnalyser(BuildingBlock):
    def __init__(self, building_block: BuildingBlock = None):
        super().__init__()
        self._input = building_block
        self._dependency_list = [building_block]
        self._histogram = Histogram()
        self._data_list = []
        self._analyse_result = {}

    def _calculate(self, syscall: Syscall):
        return self._input.get_result(syscall)

    def train_on(self, syscall: Syscall):
        inp = self._input.get_result(syscall)
        if inp is None:
            # TODO handle?
            return
        self._histogram.add(inp)
        self._data_list.append(inp)

    def fit(self):
        self._analyse_result = {
            # "variance": statistics.pvariance(self._data_list),
            # "stdev": statistics.pstdev(self._data_list),
            "entropy_e": entropy(self._data_list, base=e),
            "entropy_2": entropy(self._data_list, base=2),
            "entropy_10": entropy(self._data_list, base=10),
            "unique": len(self._histogram.keys()),
            "total": len(self._histogram)
        }

    def get_analyse_result(self) -> dict:
        return self._analyse_result

    def depends_on(self) -> list:
        return self._dependency_list


def entropy(data, base=None):
    value, counts = np.unique(data, return_counts=True)
    norm_counts = counts / counts.sum()
    base = e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()
