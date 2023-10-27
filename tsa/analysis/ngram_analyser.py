import math

import numpy as np
from numpy import e

from tsa.analysis.analyser import AnalyserBB
from tsa.utils import gini_coeff


class NgramAnalyser(AnalyserBB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree = NgramTreeNode()
        self.len = -1
        # keep track of last added n-gram because the tree won't return its sub-sequences
        self._last_ngram = None

    def _add_input(self, syscall, inp):
        if inp is None:
            # TODO
            return
        self._add_ngram(tuple(inp))

    def _add_ngram(self, ngram):
        if self.len == -1:
            self.len = len(ngram)
        elif self.len != len(ngram):
            raise ValueError("Unequal ngram size")
        self._last_ngram = ngram
        self.tree.add_ngram(ngram)

    def _iter_ngrams(self, ngram_size):
        # count the sub_ngrams for the last added n-gram because they are not covered in the tree
        sub_ngrams = {}
        if self._last_ngram is not None and len(self._last_ngram) > ngram_size:
            sub_ngrams = count_ngrams(self._last_ngram[1:], ngram_size)
        for ngram, count in self.tree.iter_length(ngram_size):
            ngram = tuple(ngram)
            if ngram in sub_ngrams:
                count += sub_ngrams[ngram]
                del sub_ngrams[ngram]
            yield ngram, count
        # additionally yield n-gram that were not yet yielded (because they only appear in the last n-gram)
        for ngram, count in sub_ngrams.items():
            yield ngram, count

    def _make_stats(self):
        stats = []
        alphabet_size = 0
        last_counts = {} # stores the n-gram counts for the last depth (used for conditional entropy calculation)
        for ngram_size in range(1, self.len + 1):
            unique = 0
            total = 0
            counts = []
            rev_cond_probs = []
            current_counts = {}
            for ngram, count in self._iter_ngrams(ngram_size):
                unique += 1
                total += count
                counts.append(count)
                current_counts[ngram] = count
                if ngram_size > 1:
                    reversed_cond_prob = last_counts[ngram[:-1]] / count
                    rev_cond_probs.append(reversed_cond_prob)
            last_counts = current_counts
            if ngram_size == 1:
                # the alphabet size is the number of unique system calls (= 1-grams)
                alphabet_size = unique
            n_possible_ngrams = math.pow(alphabet_size, ngram_size)
            density = unique / n_possible_ngrams
            counts = np.array(counts)
            rev_cond_probs = np.array(rev_cond_probs)
            norm_counts = np.array(counts) / total
            entropy = -(norm_counts * np.log(norm_counts)).sum()
            if ngram_size > 1:
                cond_entropy = (norm_counts * np.log(rev_cond_probs)).sum()
            else:
                cond_entropy = math.nan
            variability = entropy / np.log(n_possible_ngrams)
            unique_norm_entropy = entropy / np.log(unique)
            simpson_index = np.sum((counts * (counts - 1))) / (total * (total - 1))
            gini = gini_coeff(counts)
            stats.append({
                "ngram_size": ngram_size,
                "unique_ngrams": unique,
                "total": total,
                "unique_ngrams/total": unique / total,
                "entropy": entropy,
                "conditional_entropy": cond_entropy,
                "simpson_index": simpson_index,
                "gini": gini,
                "density": density,
                "variability": variability,
                "normalized_entropy": unique_norm_entropy,
                "unique_syscalls": alphabet_size
            })
        return stats


def count_ngrams(sequence, size: int):
    ngrams = {}
    for i in range(len(sequence) - size + 1):
        subsequence = sequence[i:i + size]
        ngram = tuple([t for t in subsequence])
        if ngram not in ngrams:
            ngrams[ngram] = 0
        ngrams[ngram] += 1
    return ngrams


class NgramTreeNode:
    def __init__(self):
        self._children = {}
        self._count = 0

    def add_ngram(self, ngram):
        self._count += 1
        if len(ngram) == 0:
            return

        children_part = ngram[1:]
        if ngram[0] not in self._children:
            self._children[ngram[0]] = NgramTreeNode()
        self._children[ngram[0]].add_ngram(children_part)

    def get_count(self):
        return self._count

    def get_ngram_count(self, ngram):
        if len(ngram) == 0:
            return self.get_count()
        if ngram[0] not in self._children:
            return 0
        child = self._children[ngram[0]]
        return child.get_ngram_count(ngram[1:])

    def iter_length(self, depth):
        for val, child in self._children.items():
            next_depth = depth - 1
            if next_depth < 1:
                yield [val], child.get_count()
            else:
                for suffix, count in child.iter_length(depth - 1):
                    yield [val] + suffix, count
