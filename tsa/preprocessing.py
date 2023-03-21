import math

from tqdm import tqdm

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall


class Histogram:
    def __init__(self):
        self._counts = {}

    def add(self, element):
        if element not in self._counts:
            self._counts[element] = 0
        self._counts[element] = self._counts[element] + 1

    def remove(self, element):
        if element not in self._counts:
            raise ValueError("Not in histogram: %s" % element)
        self._counts[element] -= 1
        if self._counts[element] == 0:
            del self._counts[element]

    def get_count(self, element):
        if element in self._counts:
            return self._counts[element]
        else:
            return 0

    def __iter__(self):
        for element, counts in self._counts.items():
            for _ in range(counts):
                yield element

    def __contains__(self, item):
        return item in self._counts and self._counts[item] > 0


class NgramNaiveBayes:
    def __init__(self, pseudo_count=1):
        self._prior_hist = Histogram()
        self._post_hist = Histogram()
        self._size = 0
        self._pseudo_count = pseudo_count

    def add(self, ngram):
        prior = ngram[:-1]
        self._prior_hist.add(prior)
        self._post_hist.add(ngram)
        self._size += 1

    def remove(self, ngram):
        prior = ngram[:-1]
        self._size -= 1
        self._post_hist.remove(ngram)
        self._prior_hist.remove(prior)

    def get_prob(self, ngram) -> float:
        prior = ngram[:-1]
        prior_count = self._prior_hist.get_count(prior) + self._pseudo_count
        post_count = self._post_hist.get_count(ngram) + self._pseudo_count
        return post_count / prior_count

    def sum_prob(self):
        sum_logp = 0
        for ngram in self._post_hist:
            sum_logp += math.log2(self.get_prob(ngram))
        return sum_logp

    def __contains__(self, item):
        return item in self._post_hist

    def __str__(self):
        return f"NgramNaiveBayes(size={self._size})"


class OutlierDetector(BuildingBlock):

    def __init__(self, building_block: BuildingBlock = None, lam = 0.2, c = 0.2):
        super().__init__()
        self._input = building_block
        self._dependency_list = [building_block]
        self._normal_dist = NgramNaiveBayes()
        self._anomalies_dist = NgramNaiveBayes()
        self._training_data = []
        self._lam = lam
        self._c = c

    def _ll(self):
        # TODO optimize
        return self._normal_dist._size * math.log2(
            1 - self._lam) + self._normal_dist.sum_prob() + self._anomalies_dist._size * math.log2(
            self._lam) + self._anomalies_dist.sum_prob()

    def _calculate(self, syscall: Syscall):
        ngram = self._input.get_result(syscall)
        if syscall in self._anomalies_dist:
            return None
        else:
            return ngram

    def train_on(self, syscall: Syscall):
        ngram = self._input.get_result(syscall)
        if ngram is None:
            return
        self._normal_dist.add(ngram)
        self._training_data.append(ngram)

    def fit(self):
        print("Find anomalies in training data...")
        for ngram in tqdm(self._training_data):
            normal_ll = self._ll()
            self._normal_dist.remove(ngram)
            self._anomalies_dist.add(ngram)
            anomaly_ll = self._ll()
            diff = anomaly_ll - normal_ll
            if diff <= self._c:
                # difference of likelihoods is not big enough => ngram is not an anomaly
                # move it back to normal dist
                self._normal_dist.add(ngram)
                self._anomalies_dist.remove(ngram)
        del self._training_data
        print("Anomalies:", self._anomalies_dist)
        print("Normal:", self._normal_dist)

    def depends_on(self) -> list:
        return self._dependency_list
