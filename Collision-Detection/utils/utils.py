import importlib
import itertools
from functools import partial
from typing import Callable

import numpy as np
import torch


class DeviceHelper:
    @staticmethod
    def get(config):
        if config["device"] == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config["device"] == "mps":
            return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        return torch.device("cpu")


def constrain(x, a, b):
    return np.minimum(np.maximum(x, a), b)


def not_zero(x, eps=0.01):
    if abs(x) > eps:
        return x
    elif x > 0:
        return eps
    else:
        return -eps


def wrap_to_pi(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def remap(v, x, y, clip=False):
    if x[1] == x[0]:
        return y[0]
    out = y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    if clip:
        out = constrain(out, y[0], y[1])
    return out


def pos(x):
    return np.maximum(x, 0)


def neg(x):
    return np.maximum(-x, 0)


def near_split(x, num_bins=None, size_bins=None):
    """
        Split a number into several bins with near-even distribution.

        You can either set the number of bins, or their size.
        The sum of bins always equals the total.
    :param x: number to split
    :param num_bins: number of bins
    :param size_bins: size of bins
    :return: list of bin sizes
    """
    if num_bins:
        quotient, remainder = divmod(x, num_bins)
        return [quotient + 1] * remainder + [quotient] * (num_bins - remainder)
    elif size_bins:
        return near_split(x, num_bins=int(np.ceil(x / size_bins)))


def zip_with_singletons(*args):
    """
        Zip lists and singletons by repeating singletons

        Behaves usually for lists and repeat other arguments (including other iterables such as tuples np.array!)
    :param args: arguments to zip x1, x2, .. xn
    :return: zipped tuples (x11, x21, ..., xn1), ... (x1m, x2m, ..., xnm)
    """
    return zip(
        *(arg if isinstance(arg, list) else itertools.repeat(arg) for arg in args)
    )


def class_from_path(path: str) -> Callable:
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object
