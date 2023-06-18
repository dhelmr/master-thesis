import itertools
import math
import random

from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
from tsa.unsupervised.mixed_model import Histogram


class FrequencyEncoding(BuildingBlock):
    def __init__(self, input_bb: BuildingBlock, n_components=5, threshold=20):
        super().__init__()
        self._input = input_bb
        self._counts = Histogram()
        self._mds = MDS(n_components=n_components, dissimilarity="precomputed")
        self._max_count = None
        self._embeddings = None
        self._unseen_frequency_ngram = None
        self._threshold = threshold

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
        self._max_count = self._counts.max_count()
        unseen_frequency_ngram = (-1,)
        while unseen_frequency_ngram in self._counts:
            unseen_frequency_ngram += (-1,)
        #self._counts.add(unseen_frequency_ngram)
        self._unseen_frequency_ngram = unseen_frequency_ngram
        distance_matrix = []
        for ngram in itertools.chain(self._counts.keys(), [unseen_frequency_ngram]):
            row = []
            for ngram2 in itertools.chain(self._counts.keys(), [unseen_frequency_ngram]):
                row.append(self._distance(ngram, ngram2))
            distance_matrix.append(row)
        transformed = self._mds.fit_transform(distance_matrix)
        transformed = MinMaxScaler().fit_transform(transformed)
        self._embeddings = {
            ngram: tuple(emb) for ngram, emb in zip(self._counts.keys(), transformed)
        }
        #print("embeddings", self._embeddings)
        # self._unseen_ngram_embeddings = self._determine_unseen_ngram_embeddings()
        #print("unseen embeddings:", self._unseen_ngram_embeddings)

    def _determine_unseen_ngram_embeddings(self):
        lowest_count = min(self._counts.values())
        return [emb for ngram, emb in self._embeddings.items() if self._counts.get_count(ngram) == lowest_count]

    def _distance(self, a, b):
        if self._max_count is None:
            raise ValueError("max_count is not determined yet. fit() must be called first.")
        if a == b:
            return 0
        count_a = self._counts.get_count(a)
        count_b = self._counts.get_count(b)
        if count_a > self._threshold and count_b > self._threshold:
            return 0
        return 2*self._max_count-(count_a+count_b)

    def depends_on(self) -> list:
        return [self._input]
