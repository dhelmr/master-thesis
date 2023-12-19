import itertools
import math

from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
from tsa.histogram import Histogram

ANOMALY_FUNCTIONS = [
    "linear", "homographic", "exponential", "max-scaled", "cos"
]

class FrequencyAnomalyFunction:

    def __init__(self, name, alpha):
        self._name = name
        if self._name not in ANOMALY_FUNCTIONS:
            raise ValueError("%s is not a valid anomaly function. Choose from: %s" % (name, ANOMALY_FUNCTIONS))
        if self._name == "exponential":
            if alpha <= 0 or alpha >= 1:
                raise ValueError("alpha is %s but must be in (0,1) for exponential anomaly fn" % alpha)
        elif alpha <= 0:
            raise ValueError("alpha must be > 0")

        self._alpha = alpha
        self._max_count = None

    def anomaly_value(self, ngram_frequency):
        if self._name == "linear":
            val = (self._alpha - ngram_frequency)/self._alpha
            if val < 0:
                return 0
            return val
        elif self._name == "max-scaled":
            if self._max_count is None:
                raise RuntimeError("max_count must be set first.")
            return 1 - (ngram_frequency/self._max_count)
        elif self._name == "homographic":
            return self._alpha / (ngram_frequency + self._alpha)
        elif self._name == "exponential":
            return math.pow(self._alpha, ngram_frequency)
        elif self._name == "cos":
            if ngram_frequency > self._alpha:
                return 0
            return math.cos(math.pi / (2*self._alpha) * ngram_frequency)

    def set_max_count(self, max_count):
        self._max_count = max_count

class FrequencyEncoding(BuildingBlock):
    def __init__(self, input_bb: BuildingBlock, n_components=5, threshold=None, anomaly_fn="linear", alpha=0.5, unseen_frequency=0):
        super().__init__()
        self._anomaly_fn = FrequencyAnomalyFunction(anomaly_fn, alpha)
        self._input = input_bb
        self._counts = Histogram()
        self._mds = MDS(n_components=n_components, dissimilarity="precomputed")
        self._embeddings = None
        self._unseen_frequency_ngram = None
        self._threshold = threshold
        self._unseen_frequency = unseen_frequency


    def train_on(self, syscall: Syscall):
        inp = self._input.get_result(syscall)
        if inp is None:
            return
        self._counts.add(inp)

    def _calculate(self, syscall: Syscall):
        inp = self._input.get_result(syscall)
        if inp is None:
            return
        if inp in self._embeddings:
            return inp + self._embeddings[inp]
        else:
            return inp + self._embeddings[self._unseen_frequency_ngram]

    def fit(self):
        self._anomaly_fn.set_max_count(self._counts.max_count())
        unseen_frequency_ngram = (-1,)
        while unseen_frequency_ngram in self._counts:
            unseen_frequency_ngram += (-1,)
        self._unseen_frequency_ngram = unseen_frequency_ngram
        if self._unseen_frequency > 0:
            self._counts.add(unseen_frequency_ngram, self._unseen_frequency)
        def iter_ngrams():
            if self._unseen_frequency > 0:
                return self._counts.keys()
            else:
                return itertools.chain(self._counts.keys(), [unseen_frequency_ngram])
        distance_matrix = []
        for ngram in iter_ngrams():
            row = []
            for ngram2 in iter_ngrams():
                row.append(self._distance(ngram, ngram2))
            distance_matrix.append(row)
        transformed = self._mds.fit_transform(distance_matrix)
        transformed = MinMaxScaler().fit_transform(transformed)
        self._embeddings = {
            ngram: tuple(emb) for ngram, emb in zip(iter_ngrams(), transformed)
        }
        #print("embeddings", self._embeddings)
        # self._unseen_ngram_embeddings = self._determine_unseen_ngram_embeddings()
        #print("unseen embeddings:", self._unseen_ngram_embeddings)

    def _determine_unseen_ngram_embeddings(self):
        lowest_count = min(self._counts.values())
        return [emb for ngram, emb in self._embeddings.items() if self._counts.get_count(ngram) == lowest_count]

    def _distance(self, a, b):
        if a == b:
            return 0
        count_a = self._counts.get_count(a)
        count_b = self._counts.get_count(b)
        if self._threshold is not None and (count_a > self._threshold and count_b > self._threshold):
            return 0
        return self._anomaly_fn.anomaly_value(count_a)+self._anomaly_fn.anomaly_value(count_b)

    def depends_on(self) -> list:
        return [self._input]
