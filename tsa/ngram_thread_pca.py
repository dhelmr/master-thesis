from copy import deepcopy
from typing import Dict

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from algorithms.building_block import BuildingBlock
from tsa.NgramThreadEntropy import Ngram
from tsa.frequency_encoding import FrequencyAnomalyFunction
from tsa.unsupervised.mixed_model import Histogram
from tsa.unsupervised.thread_clustering import plot


class NgramThreadMatrix:
    def __init__(self):
        self._thread_distributions: Dict[Ngram, Histogram] = {}
        self._observed_thread_ids = set()

    def add(self, ngram, thread_id):
        if ngram not in self._thread_distributions:
            self._thread_distributions[ngram] = Histogram()
        self._thread_distributions[ngram].add(thread_id)
        self._observed_thread_ids.add(thread_id)

    def ngram_thread_matrix(self):
        matrix = []
        row_labels = []
        column_labels = list(sorted(self._observed_thread_ids))
        for ngram, thread_dist in self._thread_distributions.items():
            row = [thread_dist.get_count(thread_id) for thread_id in column_labels]
            matrix.append(row)
            row_labels.append(ngram)
        return matrix, row_labels, column_labels

class NgramThreadEmbeddings:

    def __init__(self, matrix, ngrams, n_components=10):
        matrix = deepcopy(matrix)

        unseen_ngram = (-1,)
        while unseen_ngram in ngrams:
            unseen_ngram += (-1,)
        self._unseen_ngram = unseen_ngram
        # append vector for unseen ngram (that does not occur in any thread)
        null_vector = [0]*len(matrix[0])
        #matrix.append(null_vector)

        pca = PCA(n_components=n_components)
        min_max = MinMaxScaler()
        reduced = pca.fit_transform(matrix)
        reduced = min_max.fit_transform(reduced)
        plot(reduced)
        print(reduced)
        self._embeddings = {}
        for i, ngram in enumerate(ngrams):
            self._embeddings[ngram] = tuple(reduced[i])
        #self._embeddings[unseen_ngram] = tuple(reduced[-1])
        self._unseen_emb = tuple([0]*n_components)

    def get_embedding(self, ngram):
        if ngram in self._embeddings:
            return self._embeddings[ngram]

        #return self._embeddings[self._unseen_ngram]
        return self._unseen_emb

class NgramThreadEmbeddingBB(BuildingBlock):
    """
    """

    def __init__(self, input: BuildingBlock, n_components = 10, append=False):
        super().__init__()
        # parameter
        self._input = input

        # internal data
        self._ngram_thread_matrix = NgramThreadMatrix()
        self._embeddings: NgramThreadEmbeddings = None
        self._n_components = n_components
        self._append = append

        # dependency list
        self._dependency_list = []
        self._dependency_list.append(self._input)


    def depends_on(self):
        return self._dependency_list

    def train_on(self, syscall):
        """
        creates a set for distinct ngrams from training data
        """
        ngram = self._input.get_result(syscall)
        if ngram is None:
            return
        self._ngram_thread_matrix.add(ngram, syscall.thread_id())

    def fit(self):
        matrix, ngrams, threads = self._ngram_thread_matrix.ngram_thread_matrix()
        self._embeddings = NgramThreadEmbeddings(matrix, ngrams, n_components=self._n_components)

    def _calculate(self, syscall):
        inp = self._input.get_result(syscall)
        if inp is None:
            return
        emb = self._embeddings.get_embedding(inp)
        if self._append:
            return tuple(inp) + emb
        else:
            return emb

def get_ngrams(l, window_length):
    length = len(l)
    if window_length > length:
        raise ValueError("window_length must be < length")
    #for i in range()