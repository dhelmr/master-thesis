from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
from tsa.frequency_encoding import FrequencyAnomalyFunction
from tsa.unsupervised.mixed_model import Histogram


class FrequencySTIDE(BuildingBlock):
    """
    """

    def __init__(self, input: BuildingBlock, alpha=0.5, anomaly_fn = "exponential"):
        super().__init__()
        # parameter
        self._input = input

        # internal data
        self._normal_counts = Histogram()
        self._alpha = alpha

        # dependency list
        self._dependency_list = []
        self._dependency_list.append(self._input)

        self.anomaly_fn = FrequencyAnomalyFunction(anomaly_fn, alpha)

    def depends_on(self):
        return self._dependency_list

    def train_on(self, syscall: Syscall):
        """
        creates a set for distinct ngrams from training data
        """
        ngram = self._input.get_result(syscall)
        if ngram is None:
            return
        self._normal_counts.add(ngram)

    def fit(self):
        self.anomaly_fn.set_max_count(self._normal_counts.max_count())

    def _calculate(self, syscall: Syscall):
        """
        calculates ratio of unknown ngrams in sliding window of current recording
        """
        ngram = self._input.get_result(syscall)
        if ngram is None:
            return None

        if ngram in self._normal_counts:
            ngram_freq = self._normal_counts.get_count(ngram)
        else:
            ngram_freq = 0
        return self.anomaly_fn.anomaly_value(ngram_freq)

