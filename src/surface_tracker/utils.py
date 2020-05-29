"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs
Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See LICENSE for license details.
---------------------------------------------------------------------------~(*)
"""


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
