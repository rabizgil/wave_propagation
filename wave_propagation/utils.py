import math
from fractions import Fraction
from functools import reduce

from numpy import ndarray


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


def find_common_multiplier(floats):
    fractions = [Fraction(f).limit_denominator() for f in floats]
    denominators = [frac.denominator for frac in fractions]
    common_multiplier = reduce(lcm, denominators)
    return common_multiplier


def resample_arrayay(array: ndarray, new_shape: tuple | list) -> ndarray:
    shape = (
        new_shape[0],
        array.shape[0] // new_shape[0],
        new_shape[1],
        array.shape[1] // new_shape[1],
    )
    return array.reshape(shape).mean(-1).mean(1)
