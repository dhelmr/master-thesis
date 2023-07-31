from algorithms.building_block import BuildingBlock
from tsa.frequency_encoding import FrequencyAnomalyFunction
from tsa.histogram import Histogram
from tsa.ngram_thread_matrix import process_thread_id


class NgramFrequencyAppender(BuildingBlock):
    """
    """

    def __init__(self, input: BuildingBlock, anomaly_fn="max-scaled", alpha=0.5, features=None):
        super().__init__()
        if features is None:
            self._features = ["ngram_frequency", "thread_frequency"]
        else:
            self._features = features
        # parameter
        self._input = input

        # internal data
        self._normal_counts = Histogram()
        self._thread_counts = dict()

        # dependency list
        self._dependency_list = []
        self._dependency_list.append(self._input)

        self._anomaly_function = FrequencyAnomalyFunction(anomaly_fn, alpha)
        self._thread_af = FrequencyAnomalyFunction(anomaly_fn, alpha)
        self._observed_threads = set()

    def depends_on(self):
        return self._dependency_list

    def train_on(self, syscall):
        """
        creates a set for distinct ngrams from training data
        """
        ngram = self._input.get_result(syscall)
        if ngram is None:
            return
        self._normal_counts.add(ngram)
        # TODO: refactor code
        if ngram not in self._thread_counts:
            self._thread_counts[ngram] = Histogram()
        thread_id = process_thread_id(syscall)
        self._thread_counts[ngram].add(thread_id)
        self._observed_threads.add(thread_id)

    def fit(self):
        self._anomaly_function.set_max_count(self._normal_counts.max_count())
        self._thread_af.set_max_count(len(self._observed_threads))
    def _calculate(self, syscall):
        inp = self._input.get_result(syscall)
        if inp is None:
            return
        concatenated = inp
        if "ngram_frequency" in self._features:
            count = self._normal_counts.get_count(inp)
            score = self._anomaly_function.anomaly_value(count)
            concatenated = concatenated + (score,)
        if "thread_frequency" in self._features:
            if inp in self._thread_counts:
                dist = self._thread_counts[inp]
                count = len(dist.keys())
            else:
                count = 0
            score = self._anomaly_function.anomaly_value(count)
            concatenated = concatenated + (score,)
        return concatenated