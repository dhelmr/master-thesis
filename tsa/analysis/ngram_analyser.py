import numpy as np
from numpy import e

from tsa.analysis.analyser import AnalyserBB


class NgramAnalyser(AnalyserBB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree = NgramTreeNode()
        self.len = -1

    def _add_input(self, inp):
        if inp is None:
            # TODO
            return
        if self.len == -1:
            self.len = len(inp)
        elif self.len != len(inp):
            raise ValueError("Unequal ngram size")
        inp = tuple(reversed(inp))
        self.tree.add_ngram(inp)

    def _make_stats(self):
        stats = []
        for depth in range(1, self.len):
            unique = 0
            total = 0
            counts = []
            for ngram, count in self.tree.iter_length(depth):
                unique += 1
                total += count
                counts.append(count)
                print(ngram, count)
            counts = np.array(counts)
            norm_counts = np.array(counts) / total
            entropy = -(norm_counts * np.log(norm_counts) / np.log(e)).sum()
            simpson_index = np.sum((counts * (counts-1))) / ( total*(total-1))
            stats.append({
                "ngram_size": depth,
                "unique": unique,
                "total": total,
                "u/t": unique / total,
                "entropy": entropy,
                "simpson_index": simpson_index
            })
        return stats


class NgramTreeNode:
    def __init__(self):
        self._children = {}
        self._count = 0

    def add_ngram(self, ngram):
        self._count += 1

        children_part = ngram[1:]
        if len(children_part) == 0:
            return

        if ngram[0] not in self._children:
            self._children[ngram[0]] = NgramTreeNode()
        self._children[ngram[0]].add_ngram(children_part)

    def get_count(self):
        return self._count

    def iter_length(self, depth):
        for val, child in self._children.items():
            next_depth = depth -1
            if next_depth < 1:
                yield [val], child.get_count()
            else:
                for suffix, count in child.iter_length(depth-1):
                    yield [val] + suffix, count
