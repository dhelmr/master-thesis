import random
from collections import deque
from statistics import mean, median

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
from tsa.frequency_encoding import FrequencyAnomalyFunction
from tsa.histogram import Histogram
from tsa.ngram_thread_matrix import process_thread_id

AGGREGATIONS = ["mean", "median"]
class MicroSTIDEs(BuildingBlock):

    def __init__(self, input: BuildingBlock, n_models=20, models_per_trace=1, stream_sum_window_len = 1000, aggregation="mean"):
        super().__init__()
        # parameter
        self._input = input

        # internal data
        self._n_models = n_models
        self._normal_models = [set() for _ in range(n_models)]
        self._trace_ids = {}
        self._models_per_trace = models_per_trace

        self._stream_sums = [deque(maxlen=stream_sum_window_len) for _ in range(n_models)]

        self.aggregation = aggregation
        if aggregation not in AGGREGATIONS:
            raise ValueError("Invalid aggregation method: %s" % aggregation)

        # dependency list
        self._dependency_list = []
        self._dependency_list.append(self._input)

    def depends_on(self):
        return self._dependency_list

    def _get_models(self, syscall):
        trace_id = process_thread_id(syscall)
        if trace_id not in self._trace_ids:
            self._trace_ids[trace_id] = random.choices(range(self._n_models), k=self._models_per_trace)
        for model_id in self._trace_ids[trace_id]:
            yield self._normal_models[model_id]

    def train_on(self, syscall: Syscall):
        """
        creates a set for distinct ngrams from training data
        """
        ngram = self._input.get_result(syscall)
        if ngram is None:
            return
        for model in self._get_models(syscall):
            model.add(ngram)

    def fit(self):
        pass

    def _calculate(self, syscall: Syscall):
        """
        calculates ratio of unknown ngrams in sliding window of current recording
        """
        ngram = self._input.get_result(syscall)
        if ngram is None:
            return None

        sum_values = []
        for model_id in range(self._n_models):
            model = self._normal_models[model_id]
            if ngram in model:
                anomaly_value = 0
            else:
                anomaly_value = 1
            stream_sum = self._stream_sums[model_id]
            stream_sum.append(anomaly_value)
            sum_values.append(sum(stream_sum))
        if self.aggregation == "mean":
            return mean(sum_values)
        if self.aggregation == "median":
            return median(sum_values)
