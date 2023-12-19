import math

import numpy as np
import scipy
from numpy import e, nan

from tsa.utils import gini_coeff


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

    def normalized_entropy(self, n_classes=None, base=None):
        base = e if base is None else base
        n_classes = len(self.keys()) if n_classes is None else n_classes
        entropy = self.entropy(base)
        max_entr = math.log(n_classes, base)
        return entropy / max_entr

    def simpson_index(self):
        arr = self.counts_as_np_arr()
        if self._size == 1:
            return nan  # TODO!!
        return np.sum((arr * (arr - 1))) / (self._size * (self._size - 1))

    def gini_coeff(self):
        arr = self.counts_as_np_arr()
        return gini_coeff(arr)

    def fit_zipf(self, a_upper_bound=20):
        sorted_counts = sorted(self._counts.values(), reverse=True)
        ##data = list(enumerate(sorted_counts))
        fitted = scipy.stats.fit(scipy.stats.zipf, sorted_counts, bounds={"a": (0, a_upper_bound)})
        return {
            "a": fitted.params.a,
            "loc": fitted.params.loc
        }

    def values(self):
        return self._counts.values()

    def zip(self, hist2: "Histogram"):
        keys_in_hist2 = set(hist2.keys())
        for key in self.keys():
            if key in keys_in_hist2:
                keys_in_hist2.remove(key)
            yield key, self.get_count(key), hist2.get_count(key)
        for key in keys_in_hist2:
            yield key, self.get_count(key), hist2.get_count(key)

    def jensen_shannon_divergence(self, hist2: "Histogram", base=2):
        # see https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
        mixture_entropy = 0
        total1 = len(self)
        total2 = len(hist2)
        for _, c1, c2 in self.zip(hist2):
            mixture_prob = 0.5 * (c1 / total1 + c2 / total2)
            mixture_entropy += mixture_prob * math.log(mixture_prob, base)
        jsd = - mixture_entropy - 0.5 * (self.entropy(base=base) + hist2.entropy(base=base))
        if jsd < 0:  # handle numerical inaccuracies; jsd should always be >= 0
            return 0
        return jsd

    def binary_jaccard(self, hist2: "Histogram"):
        and_sum = 0
        or_sum = 0
        for _, c1, c2 in self.zip(hist2):
            or_sum += 1
            if c1 > 0 and c2 > 0:
                and_sum += 1
        return and_sum / or_sum

    def cosine_similarity(self, hist2: "Histogram"):
        dot_product = self.dot_product(hist2)
        return dot_product / (self.l2norm() * hist2.l2norm())

    def dot_product(self, hist2: "Histogram"):
        sum = 0
        for _, c1, c2 in self.zip(hist2):
            sum += c1 * c2
        return sum

    def l2norm(self):
        sum = 0
        for c in self._counts.values():
            sum += c * c
        return math.sqrt(sum)

    def bhattacharyya_coef(self, hist2):
        hist1 = self
        total1 = len(hist1)
        total2 = len(hist2)
        sum = 0
        for key, count1, count2 in self.zip(hist2):
            sum += math.sqrt(count1 / total1 * count2 / total2)
        # handle numeric inaccuracies (bc must be in [0,1])
        if sum > 1:
            return 1
        if sum < 0:
            return 0
        return sum

    def hellinger_distance(self, hist2: "Histogram"):
        bhattacharyya_coef = self.bhattacharyya_coef(hist2)
        return math.sqrt(1 - bhattacharyya_coef)

    def copy(self) -> "Histogram":
        copy_hist = Histogram()
        copy_hist._counts = self._counts.copy()
        copy_hist._size = self._size
        copy_hist._unique_elements = self._unique_elements
        return copy_hist
