import itertools
import math
import random


def split_list(l, fraction_sublist1: float):
    if fraction_sublist1 < 0 or fraction_sublist1 > 1:
        raise ValueError("Argument fraction_sublist1 must be between 0 and 1, but is: %s" % fraction_sublist1)
    size = len(l)
    split_at = math.floor(fraction_sublist1 * size)
    return l[:split_at], l[split_at:]


def random_permutation(items, k, nth_element, random_seed=0):
    items = sorted(items)
    random.Random(random_seed).shuffle(items)
    permutations = itertools.permutations(items)
    perm = choose_element(permutations, nth_element)
    return perm[:k]


def choose_element(iterator, index):
    if index < 0:
        raise ValueError("index must be >= 0")
    total = 0
    for i, el in enumerate(iterator):
        total += 1
        if i == index:
            return el
    raise ValueError(
        "Cannot choose the %s-th element, because the iterator returned only %s elements." % (index, total))
