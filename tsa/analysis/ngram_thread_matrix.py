import math
from typing import Dict, List

from dataloader.syscall import Syscall
from tsa.histogram import Histogram

Ngram = tuple
class NgramThreadMatrix:
    def __init__(self):
        self._thread_distributions: Dict[Ngram, Histogram] = {}
        self._ngram_distributions: Dict[str, Histogram] = {}
        self.total_ngrams = 0

    def add(self, ngram, thread_id):
        self.total_ngrams += 1
        if ngram not in self._thread_distributions:
            self._thread_distributions[ngram] = Histogram()
        self._thread_distributions[ngram].add(thread_id)
        if thread_id not in self._ngram_distributions:
            self._ngram_distributions[thread_id] = Histogram()
        self._ngram_distributions[thread_id].add(ngram)

    def ngram_thread_matrix(self):
        matrix = []
        row_labels = []
        column_labels = self.threads()
        for ngram, thread_dist in self._thread_distributions.items():
            row = [thread_dist.get_count(thread_id) for thread_id in column_labels]
            matrix.append(row)
            row_labels.append(ngram)
        return matrix, row_labels, column_labels

    def distributions_by_ngram(self) -> Dict[object, Histogram]:
        return self._thread_distributions

    def distributions_by_threads(self) -> Dict[str, Histogram]:
        return self._ngram_distributions

    def ngram_distances(self):
        return make_distance_matrix(self._thread_distributions)

    def thread_distances(self):
        return make_distance_matrix(self._ngram_distributions)
    def threads(self) -> List[str]:
        observed_threads = self._ngram_distributions.keys()
        return list(observed_threads)

    def ngrams(self) -> List[Ngram]:
        return list(self._thread_distributions.keys())

    def idf(self, ngram):
        n_threads = len(self._ngram_distributions.keys())
        return math.log(n_threads/len(self._thread_distributions[ngram])+1)

    def tf(self, ngram, thread_id):
        return self._ngram_distributions[thread_id].get_count(ngram)

    def tf_idf_matrix(self):
        matrix = []
        row_labels = []
        column_labels = self.threads()
        for ngram in self.ngrams():
            idf = self.idf(ngram)
            row = [idf * self.tf(ngram, thread_id) for thread_id in column_labels]
            matrix.append(row)
            row_labels.append(ngram)
        return matrix, row_labels, column_labels


def make_distance_matrix(d: Dict[object, Histogram], distance = "hellinger"):
    distance_matrix = []
    calculated_distances = {}  # used to ensure symetry of the distance matrix
    for hist1 in d.values():
        row = []
        for hist2 in d.values():
            if (hist2, hist1) in calculated_distances:
                # the distance between this pair has already been calculated
                dist = calculated_distances[(hist2, hist1)]
            else:
                dist = hist_distance(hist1, hist2, distance)
                calculated_distances[(hist1, hist2)] = dist
            row.append(dist)
        distance_matrix.append(row)
    del calculated_distances
    return distance_matrix

def hist_distance(hist1, hist2, distance_name):
    if distance_name == "jaccard-cosine":
        cos_sim = hist1.cosine_similarity(hist2)
        jaccard_sim = hist1.binary_jaccard(hist2)
        return (1-cos_sim)*(1-jaccard_sim)
    if distance_name == "cosine":
        return 1-hist1.cosine_similarity(hist2)
    if distance_name == "jaccard":
        return 1-hist1.binary_jaccard(hist2)
    elif distance_name == "hellinger":
        return hist1.hellinger_distance(hist2)
    elif distance_name == "jaccard-hellinger":
        jaccard = hist1.binary_jaccard(hist2)
        return (1-jaccard)*hist1.hellinger_distance(hist2)
    elif distance_name == "jsd":
        # jensen shannon distance is the square root of jensen shannon divergence
        return math.sqrt(hist1.jensen_shannon_divergence(hist2))
    else:
        raise ValueError("Unknown distance: %s" % distance_name)

def process_thread_id(syscall: Syscall):
    return f"({syscall.process_id()},{syscall.thread_id()})"


