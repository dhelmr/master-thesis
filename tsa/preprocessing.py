import math

from tqdm import tqdm

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall


class Histogram:
    def __init__(self):
        self._counts = {}
        self._size = 0

    def add(self, element):
        if element not in self._counts:
            self._counts[element] = 0
        self._counts[element] = self._counts[element] + 1
        self._size += 1

    def remove(self, element):
        if element not in self._counts:
            raise ValueError("Not in histogram: %s" % element)
        self._counts[element] -= 1
        if self._counts[element] == 0:
            del self._counts[element]
        self._size -= 1

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

    def __str__(self):
        return self._counts.__str__()

    def __len__(self):
        return self._size

    def keys(self):
        return self._counts.keys()

class NgramNaiveBayes:
    def __init__(self, pseudo_count=1):
        self._prior_hist = Histogram()
        self._post_hist = Histogram()
        self._size = 0
        self._pseudo_count = pseudo_count
        self._logprob_cache = {}

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
        del self._logprob_cache[prior]

    def get_log_prob(self, ngram) -> float:
        prior = ngram[:-1]
        if prior in self._logprob_cache and ngram in self._logprob_cache[prior]:
            return self._logprob_cache[prior][ngram]
        prior_count = self._prior_hist.get_count(prior) + self._pseudo_count
        post_count = self._post_hist.get_count(ngram) + self._pseudo_count
        logprob = math.log2(post_count / prior_count)
        if prior not in self._logprob_cache:
            self._logprob_cache[prior] = {}
        self._logprob_cache[prior][ngram] = logprob
        return logprob

    def sum_prob(self):
        sum_logp = 0
        for ngram in self._post_hist:
            sum_logp += self.get_log_prob(ngram)
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
        self._anomalies = {}
        self._training_data = []
        self._lam = lam
        self._c = c
        self._log_lam = math.log2(self._lam)
        self._log_reverse_lam = math.log2(1 - self._lam)

    def _ll(self, anomaly_length = None):
        if anomaly_length is None:
            anomaly_length = len(self._anomalies)
        logp_anomaly = math.log2(1/len(self._training_data))
        ll_anomalies = anomaly_length * logp_anomaly
        return self._normal_dist._size * self._log_reverse_lam + self._normal_dist.sum_prob() + anomaly_length * self._log_lam + ll_anomalies

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
            anomaly_ll = self._ll(anomaly_length=len(self._anomalies)+1)
            diff = anomaly_ll - normal_ll
            if diff <= self._c:
                # difference of likelihoods is not big enough => ngram is not an anomaly
                # move it back to normal dist
                self._normal_dist.add(ngram)
            else:
                self._anomalies.append(ngram)
        del self._training_data
        print("Anomalies:", self._anomalies)
        print("Normal:", self._normal_dist)

    def depends_on(self) -> list:
        return self._dependency_list
