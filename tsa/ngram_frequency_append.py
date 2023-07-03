from algorithms.building_block import BuildingBlock
from tsa.frequency_encoding import FrequencyAnomalyFunction
from tsa.unsupervised.mixed_model import Histogram


class NgramFrequencyAppender(BuildingBlock):
    """
    """

    def __init__(self, input: BuildingBlock, anomaly_fn="max-scaled", alpha=0.5):
        super().__init__()
        # parameter
        self._input = input

        # internal data
        self._normal_counts = Histogram()

        # dependency list
        self._dependency_list = []
        self._dependency_list.append(self._input)

        self._anomaly_function = FrequencyAnomalyFunction(anomaly_fn, alpha)

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

    def fit(self):
        self._anomaly_function.set_max_count(self._normal_counts.max_count())
    def _calculate(self, syscall):
        inp = self._input.get_result(syscall)
        if inp is None:
            return
        count = self._normal_counts.get_count(inp)
        score = self._anomaly_function.anomaly_value(count)
        return inp + (score, )