import itertools
import math
import random


def split_list(l, fraction_sublist1: float):
    if fraction_sublist1 < 0 or fraction_sublist1 > 1:
        raise ValueError("Argument fraction_sublist1 must be between 0 and 1, but is: %s" % fraction_sublist1)
    size = len(l)
    split_at = math.floor(fraction_sublist1 * size)
    return l[:split_at], l[split_at:]


def random_permutation(items, k, perm_i, random_seed=0, batch_size=10):
    if perm_i < 0:
        raise ValueError("nth_element must be >= 0")
    items = sorted(items)
    random.Random(random_seed).shuffle(items)
    a = 0
    if (int(len(items)/batch_size)) != 0:
        a = int(perm_i / int(len(items)/batch_size))
    list_offset = (perm_i * batch_size + a) % len(items)
    items = set_list_offset(items, list_offset)
    permutations = itertools.permutations(items)
    perm = choose_element(permutations, perm_i)
    return perm[:k]


#
# def random_permutation(items, k, perm_i, random_seed=0, batch_size=10):
#     if perm_i < 0:
#         raise ValueError("nth_element must be >= 0")
#     items = sorted(items)
#     for i in range(int(len(items)/(perm_i+1))+1):
#         random.Random(random_seed).shuffle(items)
#     list_offset = get_offset(len(items), batch_size, perm_i % len(items))
#     items = set_list_offset(items, list_offset)
#     #permutations = itertools.permutations(items)
#     #perm = choose_element(permutations, perm_i)
#     return tuple(items[:k])

# def get_offset(length: int, step_size: int, i: int):
#     if i >= length:
#         raise ValueError("i must be < length")
#     offset = 0
#     selected_offsets = {0}
#     while len(selected_offsets) < i:
#         offset += step_size
#         if offset >= length:
#             offset = step_size
#         while offset in selected_offsets:
#             offset += 1
#             if offset > length:
#                 offset = 0
#         selected_offsets.add(offset)
#     return offset
#
#
#     return offset



# def random_permutation(items, k, perm_i, random_seed=0, batch_size=10):
#     if k > batch_size:
#         raise ValueError("k must be <= batch_size")
#     r = random.Random(random_seed)
#     items = sorted(items)
#     free_indexes = [i for i, _ in enumerate(items)]
#     for i in range(perm_i+1):
#         selected_subset = []
#         for j in range(batch_size):
#             next_i = r.choice(free_indexes)
#             selected_subset.append(items[next_i])
#             free_indexes.remove(next_i)
#             if len(free_indexes) == 0:
#                 free_indexes = [i for i, _ in enumerate(items)]
#     return tuple(selected_subset[:k])

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
