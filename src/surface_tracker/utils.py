import functools
import typing as T
import warnings

import numpy as np
import cv2


def left_rotation(a: list, k: int):
    """Rotate list to the left
    e.g.: [1, 2, 3, 4] -> [2, 3, 4, 1]
    """
    # if the size of k > len(a), rotate only necessary with
    # # module of the division
    rotations = k % len(a)
    return a[rotations:] + a[:rotations]


def right_rotation(a: list, k: int):
    """Rotate list to the right
    e.g.: [1, 2, 3, 4] -> [4, 1, 2, 3]
    """
    # if the size of k > len(a), rotate only necessary with
    # module of the division
    rotations = k % len(a)
    return a[-rotations:] + a[:-rotations]
