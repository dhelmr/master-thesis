import abc
import argparse
import random
from abc import ABC

import numpy
import numpy as np
import pandas

from tsa.perf_pred.cv import PerformancePredictor


class Heuristic(PerformancePredictor, ABC):
    def train(self, train_X: pandas.DataFrame, train_y: numpy.ndarray):
        pass  # do nothing

    def predict(self, test_X: pandas.DataFrame) -> numpy.ndarray:
        preds = test_X.apply(self._decide_row, axis=1)
        return np.array(preds)

    @abc.abstractmethod
    def _decide_row(self, row) -> int:
        pass


class BaselineRandom(PerformancePredictor):

    def train(self, train_X: pandas.DataFrame, train_y: numpy.ndarray):
        pass

    def predict(self, test_X: pandas.DataFrame) -> numpy.ndarray:
        preds = [random.randint(0, 1) for _ in range(len(test_X))]
        return numpy.array(preds)


class BaselineAlways0(Heuristic):

    def _decide_row(self, row) -> int:
        return 0


class BaselineAlways1(Heuristic):

    def _decide_row(self, row) -> int:
        return 1


class BaselineMajorityClass(Heuristic):
    def train(self, train_X: pandas.DataFrame, train_y: numpy.ndarray):
        self.ones = 0
        self.zeros = 0
        for c in train_y:
            if c == 0:
                self.ones += 1
            elif c == 1:
                self.zeros += 1
            else:
                raise ValueError("Unexpected value: %s" % c)

    def _decide_row(self, row) -> int:
        total = self.zeros + self.ones
        return numpy.random.choice(numpy.arange(0, 2), p=[self.zeros/total, self.ones/total])


class Heuristic1(Heuristic):
    def _decide_row(self, row) -> int:
        if row["unique_ngrams/n_threads"] < 0.75 and 0.45 < row["ngram_dists_norm_entropy_mean"] < 0.6:
            return 1
        else:
            return 0


class Heuristic2(Heuristic):
    def _decide_row(self, row) -> int:
        if row["unique_ngrams/total"] < 0.0003 and row["unique_ngrams/n_threads"] < 0.75:
            return 1
        else:
            return 0

class Threshold(PerformancePredictor):
    def __init__(self, cli_args=[]):
        super().__init__(cli_args)
        parser = argparse.ArgumentParser()
        parser.add_argument("--threshold-feature", type=str, required=True)
        parser.add_argument("--threshold-lt", action="store_true", required=True)
        parser.add_argument("--threshold-value", type=float, required=True)
        # parser.add_argument("--min-samples-leaf", default=3, type=int)
        args = parser.parse_args(cli_args)
        self.clf_args = vars(args)

    def train(self, train_X: pandas.DataFrame, train_y: numpy.ndarray):
        pass  # do nothing
    def predict(self, test_X: pandas.DataFrame) -> numpy.ndarray:
        preds = test_X.apply(self._decide_row, axis=1)
        return np.array(preds)

    def _decide_row(self, row):
        if self.clf_args["threshold_lt"]:
            return 1 if row[self.clf_args["threshold_feature"]] < self.clf_args["threshold_value"] else 0
        else:
            return 1 if row[self.clf_args["threshold_feature"]] >= self.clf_args["threshold_value"] else 0
