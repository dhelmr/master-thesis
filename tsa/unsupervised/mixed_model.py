import math

import numpy as np
from numpy import e
from tqdm import tqdm

from algorithms.building_block import BuildingBlock
from tsa.unsupervised.preprocessing import OutlierDetector


class Histogram:
    def __init__(self):
        self._counts = {}
        self._size = 0
        self._unique_elements = 0

    def add(self, element, count=1):
        if count < 1:
            raise ValueError(f"Invalid count {count}. Must be >= 1.")
        if element not in self._counts:
            self._counts[element] = 0
            self._unique_elements += 1
        self._counts[element] = self._counts[element] + count
        self._size += count

    def remove(self, element):
        if element not in self._counts:
            raise ValueError(f"Not in histogram: {element}")
        self._counts[element] -= 1
        if self._counts[element] == 0:
            del self._counts[element]
            self._unique_elements -= 1
        self._size -= 1

    def get_count(self, element):
        if element in self._counts:
            return self._counts[element]
        else:
            return 0

    def __iter__(self):
        return iter(self._counts.items())

    def __contains__(self, item):
        return item in self._counts and self._counts[item] > 0

    def __str__(self):
        return self._counts.__str__()

    def __len__(self):
        return self._size

    def keys(self):
        return self._counts.keys()

    def max_count(self):
        return max(self._counts.values())

    def count_frequencies(self):
        frequencies = {}
        for count in self._counts.values():
            if count not in frequencies:
                frequencies[count] = 0
            frequencies[count] += 1
        return frequencies

    def remove_all(self, element):
        if element not in self._counts:
            raise ValueError(f"Not in histogram: {element}")
        self._size -= self._counts[element]
        del self._counts[element]
        self._unique_elements -= 1

    def reduce(self, element, reduce_by: int):
        if element not in self._counts:
            raise ValueError(f"Not in histogram: {element}")
        current = self.get_count(element)
        if reduce_by > current:
            raise ValueError(
                f"Cannot reduce count for {element}. Reduce count {reduce_by} must be <= than current value {current}. ")
        self._counts[element] -= reduce_by
        self._size -= reduce_by
        if self._counts[element] == 0:
            del self._counts[element]
            self._unique_elements -= 1

    def unique_elements(self):
        return self._unique_elements

    def counts_as_np_arr(self):
        return np.array(list(self._counts.values()))

    def entropy(self, base=None):
        counts_array = self.counts_as_np_arr()
        norm_counts = counts_array / self._size
        base = e if base is None else base
        return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()

    def simpson_index(self):
        arr = self.counts_as_np_arr()
        return np.sum((arr * (arr-1))) / ( self._size*(self._size-1))

    def values(self):
        return self._counts.values()


class NgramNaiveBayes:
    def __init__(self, pseudo_count=1):
        self._prior_hist = Histogram()
        self._post_hist = Histogram()
        self._size = 0
        self._pseudo_count = pseudo_count
        self._logprob_cache = {}

    def add(self, ngram, count=1):
        if count < 1:
            raise ValueError(f"Invalid count {count}. Must be >= 1.")
        prior = ngram[:-1]
        self._prior_hist.add(prior, count)
        self._post_hist.add(ngram, count)
        self._size += count

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
        post_count = self._post_hist.get_count(ngram) + self._pseudo_count * self._post_hist.unique_elements()
        logprob = math.log2(post_count / prior_count)
        if prior not in self._logprob_cache:
            self._logprob_cache[prior] = {}
        self._logprob_cache[prior][ngram] = logprob
        return logprob

    def sum_prob(self):
        sum_logp = 0
        for ngram, count in self._post_hist:
            sum_logp += self.get_log_prob(ngram) * count
        return sum_logp

    def __contains__(self, item):
        return item in self._post_hist

    def __str__(self):
        return f"NgramNaiveBayes(size={self._size})"

    def remove_all(self, ngram) -> int:
        if ngram not in self:
            raise ValueError("ngram not found: %s" % ngram)
        prior = ngram[:-1]
        count = self._post_hist.get_count(ngram)
        self._size -= count
        self._post_hist.remove_all(ngram)
        self._prior_hist.reduce(prior, count)
        del self._logprob_cache[prior]
        return count


class MixedModelOutlierDetector(OutlierDetector):
    def __init__(self, building_block: BuildingBlock = None, train_features=None, lam=0.2, c=0.2):
        super().__init__(building_block, train_features)
        self._normal_dist = NgramNaiveBayes()
        self._num_anomalies = 0
        self._lam = lam
        self._c = c
        self._log_lam = math.log2(self._lam)
        self._log_reverse_lam = math.log2(1 - self._lam)


    def _ll(self, anomaly_length):
        logp_anomaly = math.log2(1 / len(self._training_data))
        ll_anomalies = anomaly_length * logp_anomaly
        return self._normal_dist._size * self._log_reverse_lam + self._normal_dist.sum_prob() + anomaly_length * self._log_lam + ll_anomalies

    def detect_anomalies(self, training_data):
        print("Find anomalies in training data...")
        anomalies = set()
        for ngram in training_data:
            self._normal_dist.add(ngram)
        training_data_set = set(training_data)
        for ngram in tqdm(training_data_set):
            normal_ll = self._ll(self._num_anomalies)
            self._normal_dist.remove(ngram)
            anomaly_ll = self._ll(anomaly_length=self._num_anomalies + 1)
            diff = anomaly_ll - normal_ll
            if diff <= self._c:
                # difference of likelihoods is not big enough => ngram is not an anomaly
                # move it back to normal dist
                self._normal_dist.add(ngram, 1)
            else:
                remove_count = 1
                if ngram in self._normal_dist:
                    remove_count += self._normal_dist.remove_all(ngram)
                anomalies.add(ngram)
                self._num_anomalies += remove_count
        return anomalies
