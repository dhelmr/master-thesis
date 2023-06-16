import unittest

from tsa.utils import random_permutation


class PermutationITest(unittest.TestCase):
    def test_permutation_i(self):
        items_a = [10, 11, 12, 13, 14, 15, 16]
        items_b = [11, 12, 16,15,13,14,10]
        for k in range(7):
            for i in [0,10,20,30,40,50,100,120]:
                perm_a = random_permutation(items_a, k, i)
                perm_b = random_permutation(items_b, k, i)
                self.assertEquals(perm_a, perm_b)

    def test_monotony(self):
        items = list(range(300))
        for i in range(15):
            last = ()
            for k in range(1, 20):
                perm = random_permutation(items, k, i)
                self.assertEquals(last, perm[:k-1])
                last = perm

    def test_small_list(self):
        for length in range(10,20):
            items = list(range(length))
            perms = set()
            for i in range(length):
                for k in range(1,length):
                    perm = random_permutation(items, k, i, random_seed=0)
                    self.assertNotIn(perm, perms)
                    perms.add(perm)

    def test_no_duplc(self):
        items = list(range(50))
        for seed in range(2500,2510):
            for k in range(20):
                for i in range(10):
                    perm = random_permutation(items, k, i, random_seed=seed)
                    self.assertEquals(len(perm), len(set(perm)))

    def test_empty_intersect(self):
        for l in [5,20,50]:
            items = list(range(l))
            for seed in range(2500, 2510):
                for k in range(1,l):
                    perms = set()
                    for i in range(l):
                        perm = random_permutation(items, k, i, random_seed=seed)
                        self.assertNotIn(perm, perms)
                        perms.add(perm)

    def test_unique_elements(self):
        items = list(range(450))
        selected_items = set()
        for i in range(15):
            perm = random_permutation(items, 30, i, step_size=30)
            for p in perm:
                self.assertNotIn(p, selected_items)
                selected_items.add(p)

if __name__ == "__main__":
    unittest.main()
