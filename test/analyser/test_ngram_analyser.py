import pprint
import unittest

from tsa.analysis.ngram_analyser import NgramAnalyser
from tsa.utils import random_permutation


class NgramAnalyserTest(unittest.TestCase):

    def test_ngram_analyser(self):
        analyser = NgramAnalyser(None)
        text = "abcabcaabbacadd"
        for ngram in make_ngrams(text, size=5):
            analyser._add_ngram(ngram)
        by_ngram_size = self._get_stats_by_ngram_size(analyser)
        self.assertSetEqual(set(by_ngram_size.keys()), {1, 2, 3, 4, 5})
        self.assertEqual(by_ngram_size[1]["density"], 1)
        self.assertEqual(by_ngram_size[1]["unique_ngrams"], 4)  # a,b,c
        self.assertEqual(by_ngram_size[2]["unique_ngrams"], 9)  # ab, bc, ca, aa, bb, ba, ac, ad, dd
        self.assertEqual(by_ngram_size[3]["unique_ngrams"], 11)  # abc, bca, cab, caa, aab, abb, bba, bac, aca, cad, add
        self.assertEqual(by_ngram_size[5]["unique_ngrams"], 11)

        self.assertEqual(by_ngram_size[5]["total"], 11)
        self.assertEqual(by_ngram_size[4]["total"], 12)
        self.assertEqual(by_ngram_size[1]["total"], 15)

        self.assertEqual(by_ngram_size[1]["unique_syscalls"], 4)
        self.assertEqual(by_ngram_size[4]["unique_syscalls"], 4)
        self.assertEqual(by_ngram_size[5]["unique_syscalls"], 4)

        self.assertEqual(by_ngram_size[2]["density"], 9 / 16)  # 9 out of 16 possible 2-gram appear in the sequence
        self.assertAlmostEqual(by_ngram_size[2]["entropy"], 2.0692, places=4)
        self.assertAlmostEqual(by_ngram_size[3]["entropy"], 2.35167, places=5)
        self.assertAlmostEqual(by_ngram_size[2]["normalized_entropy"], 0.941734,
                               places=5)  # entropy/max entropy for n=2: 2.0692/log(9)
        self.assertAlmostEqual(by_ngram_size[2]["variability"], 0.746306,
                               places=5)  # entropy/max entropy for n=2 (induced by alphabet size): 2.0692/log(4^2)
        self.assertAlmostEqual(by_ngram_size[2]["unique_ngrams/total"], 0.6428, places=2)
        self.assertAlmostEqual(by_ngram_size[2]["conditional_entropy"], 0.87905, places=4)

    def test_multiple_traces(self):
        analyser = NgramAnalyser(None)
        traces = [
            "abcdccdabacba",
            "ddaaaaaaaaaaa",
            "aaaaaaaaaaaaa",
            "aaaaaaaaaaaaa",
            "bbbbaaaaaaaae",
            "aaaaaaaaaaaaa",
        ]
        for i, trace in enumerate(traces):
            for ngram in make_ngrams(trace, size=5):
                analyser._add_ngram(ngram, trace_id=i)
        by_ngram_size = self._get_stats_by_ngram_size(analyser)
        self.assertEqual(by_ngram_size[1]["unique_ngrams"], 5)
        self.assertEqual(by_ngram_size[1]["unique_syscalls"], 5)
        self.assertEqual(by_ngram_size[2]["unique_syscalls"], 5)
        self.assertEqual(by_ngram_size[5]["unique_syscalls"], 5)

    def _get_stats_by_ngram_size(self, analyser):
        stats = analyser._make_stats()
        by_ngram_size = {stat["ngram_size"]: stat for stat in stats}
        return by_ngram_size

def make_ngrams(text, size: int):
    ngrams = []
    for i in range(len(text) - size + 1):
        subsequence = text[i:i + size]
        ngram_tokens = [t for t in subsequence]
        ngrams.append(tuple(ngram_tokens))
    return ngrams


if __name__ == "__main__":
    unittest.main()
