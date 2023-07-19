import math
from statistics import mean
from typing import Dict, List

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
from tsa.frequency_encoding import FrequencyAnomalyFunction
from tsa.unsupervised.mixed_model import Histogram
from tsa.unsupervised.thread_clustering import make_thread_distance_matrix

Ngram = tuple

DISTANCE_VALUES = ["hellinger"]

FEATURE_NGRAM_FREQ = "ngram_frequency"
FEATURE_THREAD = "thread_frequency"
FEATURE_NORMALIZED_ENTR = "normalized_entropy"



class NgramThreadDistance(BuildingBlock):
    """
    """

    def __init__(self, input: BuildingBlock):
        super().__init__()
        # parameter
        self._input = input

        # internal data
        self._thread_distributions: Dict[Ngram, Histogram] = {}
        self._ngram_frequencies = Histogram()
        self._observed_thread_ids = set()

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
        if ngram not in self._thread_distributions:
            self._thread_distributions[ngram] = Histogram()
        thread_id = syscall.thread_id()
        self._observed_thread_ids.add(thread_id)
        self._thread_distributions[ngram].add(thread_id)
        self._ngram_frequencies.add(ngram)

    def fit(self):
        distance_matrix = make_thread_distance_matrix(self._thread_distributions)
        for ngram, thread_dist in self._thread_distributions.items():
            normalized_entropy = thread_dist.entropy() / max_entropy
            entropy_anomaly_value = pow(1-normalized_entropy, self._entropy_alpha)/self._entropy_scale
            ngram_freq = self._ngram_frequencies.get_count(ngram)
            n_threads = len(thread_dist.keys())
            anomaly_value = self._combine_scores(freq_score=self.freq_anomaly_fn.anomaly_value(ngram_freq),
                                                 thread_freq_score=self.thread_anomaly_fn.anomaly_value(n_threads),
                                                 normalized_entropy=entropy_anomaly_value)
            self._anomaly_values[ngram] = anomaly_value

    def _combine_scores(self, freq_score, thread_freq_score, normalized_entropy):
        features = []
        if FEATURE_NGRAM_FREQ in self._features:
            features.append(freq_score)
        if FEATURE_THREAD in self._features:
            features.append(thread_freq_score)
        if FEATURE_NORMALIZED_ENTR in self._features:
            features.append(normalized_entropy)

        if self._combine == "arithmetic":
            return mean(features)
        if self._combine == "harmonic":
            denom = 0
            for f in features:
                denom += 1 / f
            return len(features) / denom
        if self._combine == "geometric":
            product = 1
            for f in features:
                product = product * f
            return math.pow(product, 1/len(features))

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
