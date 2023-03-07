import math


def split_list(l, fraction_sublist1: float):
    if fraction_sublist1 < 0 or fraction_sublist1 > 1:
        raise ValueError("Argument fraction_sublist1 must be between 0 and 1, but is: %s" % fraction_sublist1)
    size = len(l)
    split_at = math.floor(fraction_sublist1 * size)
    return l[:split_at], l[split_at:]
