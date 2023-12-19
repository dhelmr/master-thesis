from typing import Dict

import numpy
from sklearn.preprocessing import minmax_scale

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
from tsa.analysis.ngram_thread_matrix import NgramThreadMatrix, process_thread_id

Ngram = tuple

DISTANCE_VALUES = ["hellinger"]

class NgramThreadDistance(BuildingBlock):
    """
    """

    def __init__(self, input: BuildingBlock):
        super().__init__()
        # parameter
        self._input = input

        # internal data
        self._matrix = NgramThreadMatrix()

        self._anomaly_values: Dict[Ngram, float] = {}

        # dependency list
        self._dependency_list = []
        self._dependency_list.append(self._input)

    def depends_on(self):
        return self._dependency_list

    def train_on(self, syscall: Syscall):
        """
        creates a set for distinct ngrams from training data
        """
        ngram = self._input.get_result(syscall)
        if ngram is None:
            return
        self._matrix.add(ngram, process_thread_id(syscall))

    def fit(self):
        thread_distances = self._matrix.thread_distances()
        thread_anomaly_scores = self.thread_anomaly_scores(thread_distances)
        thread_anomaly_scores = minmax_scale(thread_anomaly_scores.reshape(-1,1), axis=0)
        thread_anomaly_scores = thread_anomaly_scores * 0.8
        matrix, ngrams, threads = self._matrix.ngram_thread_matrix()
        for ngram_i, row in enumerate(matrix):
            anomaly_score = 0
            total = 0
            for i, cell in enumerate(row):
                anomaly_score += cell*thread_anomaly_scores[i]
                total += cell
            anomaly_score = anomaly_score / total
            ngram = ngrams[ngram_i]
            self._anomaly_values[ngram] = anomaly_score
            print(ngram, anomaly_score)

    def thread_anomaly_scores(self, thread_distances):
        mean_distance = numpy.mean(thread_distances, axis=1)
        print(mean_distance)
        return mean_distance
    def _calculate(self, syscall: Syscall):
        """
        calculates ratio of unknown ngrams in sliding window of current recording
        """
        ngram = self._input.get_result(syscall)
        if ngram is None:
            return None

        if ngram not in self._anomaly_values:
            # ngram does not occur in training set => return anomaly value 1 by convention
            return 1
        return self._anomaly_values[ngram]
