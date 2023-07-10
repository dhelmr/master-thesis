import unittest

from tsa.cli.eval import calc_area_under_curve


class EvalTest(unittest.TestCase):
    def test_auc_calc(self):
        X = [0,1,5,10,100]
        Y = [0,0,20,100,100]
        expected = 9340 / 100
        self.assertEqual(calc_area_under_curve(X,Y), expected)

    def test_unsorted_auc_calc(self):
        X = [0,10,9,10,20]
        Y = [0,0,20,30,40]
        with self.assertRaises(ValueError) as e:
            calc_area_under_curve(X,Y)

if __name__ == "__main__":
    unittest.main()
