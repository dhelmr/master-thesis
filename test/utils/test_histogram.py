import math
import unittest
import scipy

from tsa.histogram import Histogram


class HistogramTest(unittest.TestCase):

    def test_add(self):
        hist = Histogram()
        hist.add("a", 100)
        hist.add("a")
        hist.add((1, 2))
        hist.add((1, 2), 10)
        hist.add("b")
        hist.add(None)
        self.assertEqual(hist.get_count("a"), 101)
        self.assertEqual(hist.get_count((1, 2)), 11)
        self.assertEqual(hist.get_count("b"), 1)
        self.assertEqual(hist.get_count(None), 1)
        self.assertEqual(hist.get_count("c"), 0)
        hist.remove("b")
        self.assertEqual(hist.get_count("b"), 0)
        self.assertTrue("b" not in hist)
        self.assertTrue(None in hist)
        self.assertTrue((1, 2) in hist)
        self.assertEqual(set(hist.keys()), {"a", (1, 2), None})
        self.assertEqual(hist.unique_elements(), 3)
        self.assertEqual(hist.max_count(), 101)
        self.assertEqual(len(hist), 101 + 11 + 1)

        hist.remove_all("a")
        self.assertEqual(len(hist), 12)
        self.assertTrue("a" not in hist)
        self.assertEqual(hist.get_count("a"), 0)
        self.assertEqual(hist.unique_elements(), 2)

    # def test_count_frequencies(self):
    #    hist = Histogram()
    #    for i in range(1,100):
    #        hist.add(i, count=i)
    #    for i in range(1,5):
    #        hist.add(i, count=i)
    #    counts = hist.count_frequencies()
    #    self.assertEqual(counts, [2]*10+[1]*90)

    def test_entropy(self):
        hist = Histogram()
        hist.add("a")
        self.assertEqual(hist.entropy(), 0)
        hist.add("a", 9)
        self.assertEqual(hist.entropy(), 0)
        hist.add("b", 10)
        self.assertEqual(hist.entropy(), math.log(2))
        hist.add("c", 1)
        self.assertAlmostEqual(hist.entropy(base=2), 1.2285764, places=6)

    def test_simpson(self):
        hist = Histogram()
        hist.add("a", 10)
        for i in range(1, 10):
            hist.add(i, count=1)
        self.assertEqual(len(hist), 19)
        self.assertAlmostEqual(hist.simpson_index(), 0.2632, places=4)
        self.assertAlmostEqual(hist.gini_coeff(), 0.4263, places=4)

    def test_simpson2(self):
        hist = Histogram()
        hist.add("a", 100)
        for i in range(1, 6):
            hist.add(i, count=5)
        hist.add("c", 2)
        hist.add("d", 2)
        hist.add("e", 1)
        hist.add("f", 1)
        self.assertEqual(len(hist), 131)
        self.assertAlmostEqual(hist.simpson_index(), 0.5874, places=4)
        self.assertAlmostEqual(hist.gini_coeff(), 0.7198, places=4)

    def test_jensen_shannon(self):
        a = [5, 1, 2, 5, 10, 0]
        b = [0, 4, 1, 0, 10, 3]
        hist_a = list_to_histogram(a)
        hist_b = list_to_histogram(b)
        self.assertEqual(hist_b.unique_elements(), 4)
        self.assertEqual(hist_a.unique_elements(), 5)
        for base in [2, scipy.e, 10]:
            expected_value = scipy.spatial.distance.jensenshannon(a, b, base=base)
            js_divergence = hist_a.jensen_shannon_divergence(hist_b, base=base)
            js_distance = math.sqrt(js_divergence)
            self.assertAlmostEqual(expected_value, js_distance, places=10)

    def test_cosine_sim(self):
        hist1 = list_to_histogram([0, 0, 1, 1, 10])
        self.assertAlmostEqual(hist1.cosine_similarity(list_to_histogram([0, 0, 1, 1, 10])), 1, places=10)
        self.assertAlmostEqual(hist1.cosine_similarity(list_to_histogram([0, 0, 2, 2, 20])), 1, places=10)
        self.assertAlmostEqual(hist1.cosine_similarity(list_to_histogram([10, 0, 0, 0, 20])), 0.885615, places=5)

    def test_binary_jaccard(self):
        l1 = list_to_histogram([10, 1, 0, 1, 0, 10])
        l2 = list_to_histogram([0, 1, 0, 1, 1, 10, 100, 1])
        self.assertEqual(l1.binary_jaccard(l2), 3 / 7)

    def test_hellinger(self):
        l1 = list_to_histogram([1, 1, 2, 0])
        l2 = list_to_histogram([0, 2, 2, 2])
        exp = math.sqrt(1 - (math.sqrt(1 / 4 * 0 / 6) + math.sqrt(1 / 4 * 2 / 6) + math.sqrt(2 / 4 * 2 / 6) + math.sqrt(
            0 / 4 * 2 / 6)))
        self.assertAlmostEqual(l1.hellinger_distance(l2), exp)


def list_to_histogram(l):
    hist = Histogram()
    for el, count in enumerate(l):
        if count != 0:
            hist.add(el, count)
    return hist
