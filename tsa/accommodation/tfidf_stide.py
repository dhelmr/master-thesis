import numpy as np

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
from tsa.analysis.ngram_thread_matrix import NgramThreadMatrix, process_thread_id

AGGREGATION_FUNCTIONS = {
    "mean": np.mean,
    "max": np.max
}


class TfidfSTIDE(BuildingBlock):
    """
    """

    def __init__(self, input: BuildingBlock, unseen_factor=1.5, aggregation="mean"):
        super().__init__()
        # parameter
        self._input = input

        # internal data
        self._matrix = NgramThreadMatrix()

        # dependency list
        self._dependency_list = []
        self._dependency_list.append(self._input)
        self._unseen_factor = unseen_factor
        self._aggregation = aggregation
        if aggregation not in AGGREGATION_FUNCTIONS:
            raise ValueError("%s is no valid aggregation function" % aggregation)

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
        tfidf, ngrams, threads = self._matrix.tf_idf_matrix()
        self._anomaly_scores = {}
        for i, ngram in enumerate(ngrams):
            non_zero_tfidfs = [t for t in tfidf[i] if t != 0]
            anomaly_score = AGGREGATION_FUNCTIONS[self._aggregation](non_zero_tfidfs)
            self._anomaly_scores[ngram] = anomaly_score
            print(ngram, anomaly_score)
        self._max_score = max(self._anomaly_scores.values())
        del self._matrix

    def _calculate(self, syscall: Syscall):
        """
        calculates ratio of unknown ngrams in sliding window of current recording
        """
        ngram = self._input.get_result(syscall)
        if ngram is None:
            return None

        if ngram in self._anomaly_scores:
            return self._anomaly_scores[ngram]
        else:
            return self._max_score * self._unseen_factor
