import itertools
import math
import random


def split_list(l, fraction_sublist1: float):
    if fraction_sublist1 < 0 or fraction_sublist1 > 1:
        raise ValueError("Argument fraction_sublist1 must be between 0 and 1, but is: %s" % fraction_sublist1)
    size = len(l)
    split_at = math.floor(fraction_sublist1 * size)
    return l[:split_at], l[split_at:]


def random_permutation(items, k, perm_i, random_seed=0, step_size=10):
    if perm_i < 0:
        raise ValueError("nth_element must be >= 0")
    items = sorted(items)
    random.Random(random_seed).shuffle(items)
    list_offset = get_offset(len(items), perm_i, step_size)
    items = set_list_offset(items, list_offset)
    permutations = itertools.permutations(items)
    perm = choose_element(permutations, perm_i)
    return perm[:k]


def get_offset(length, perm_i, step_size):
    offsets = list(range(length))
    selected = set([0])
    offset = 0
    for i in range(perm_i):
        offset = (offset + step_size) % len(offsets)
        while offset in selected:
            offset += 1
        selected.add(offset)
    return offset % length


def set_list_offset(l, new_offset) -> list:
    a = l[:new_offset]
    b = l[new_offset:]
    return b+a

def access_cfg(root_obj, *keys, default=None, required=True, exp_type=None):
    if default is not None:
        required = False
    cur_obj = root_obj
    cur_key = ""
    for key in keys:
        if not isinstance(cur_obj, dict) and not isinstance(cur_obj, list):
            raise ValueError("Parameter '%s' is not a dict or list." % cur_key)
        if isinstance(cur_obj, dict) and key not in cur_obj:
            if not required:
                return default
            else:
                raise ValueError("Cannot find parameter for key '%s' at %s" % (key, cur_key if cur_key != "" else "[ROOT]"))
        cur_obj = cur_obj[key]
        cur_key = "%s.%s" % (cur_key, key)
    if exp_type is not None and not isinstance(cur_obj, exp_type):
        raise ValueError("Parameter '%s' is not of expected type %s" % (cur_key, exp_type))
    return cur_obj

def exists_key(root_obj, *keys) -> bool:
    val = access_cfg(root_obj, *keys, default=None, required=False)
    return val is not None

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
