import numpy as np
from numpy import e

from tsa.analysis.analyser import AnalyserBB
from tsa.utils import gini_coeff


class NgramAnalyser(AnalyserBB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree = NgramTreeNode()
        self.len = -1

    def _add_input(self, syscall, inp):
        if inp is None:
            # TODO
            return
        if self.len == -1:
            self.len = len(inp)
        elif self.len != len(inp):
            raise ValueError("Unequal ngram size")
        # inp = tuple(reversed(inp))
        inp = tuple(inp)
        self.tree.add_ngram(inp)

    def _make_stats(self):
        stats = []
        for depth in range(1, self.len + 1):
            unique = 0
            total = 0
            counts = []
            rev_cond_probs = []
            for ngram, count in self.tree.iter_length(depth):
                unique += 1
                total += count
                counts.append(count)
                reversed_cond_prob = self.tree.get_ngram_count(ngram[:-1]) / count
                rev_cond_probs.append(reversed_cond_prob)
            counts = np.array(counts)
            rev_cond_probs = np.array(rev_cond_probs)
            norm_counts = np.array(counts) / total
            entropy = -(norm_counts * np.log(norm_counts) / np.log(e)).sum()
            cond_entropy = (norm_counts * np.log(rev_cond_probs)).sum()
            simpson_index = np.sum((counts * (counts - 1))) / (total * (total - 1))
            gini = gini_coeff(counts)
            stats.append({
                "ngram_size": depth,
                "unique": unique,
                "total": total,
                "u/t": unique / total,
                "entropy": entropy,
                "conditional_entropy": cond_entropy,
                "simpson_index": simpson_index,
                "gini": gini
            })
        return stats


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
