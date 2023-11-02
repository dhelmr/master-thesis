import math
import unittest
import scipy

from tsa.histogram import Histogram


class HistogramTest(unittest.TestCase):

    def test_jensen_shannon(self):
        a = [5,1,2,5,10,0]
        b = [0,4,1,0,10,3]
        hist_a = list_to_histogram(a)
        hist_b = list_to_histogram(b)
        self.assertEqual(hist_b.unique_elements(), 4)
        self.assertEqual(hist_a.unique_elements(), 5)
        for base in [2, scipy.e, 10]:
            expected_value = scipy.spatial.distance.jensenshannon(a, b, base=base)
            js_divergence = hist_a.jensen_shannon_divergence(hist_b, base=base)
            js_distance = math.sqrt(js_divergence)
            self.assertAlmostEqual(expected_value, js_distance, places=10)



def list_to_histogram(l):
    hist = Histogram()
    for el, count in enumerate(l):
        if count != 0:
            hist.add(el, count)
    return hist