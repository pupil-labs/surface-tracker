"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs
Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See LICENSE for license details.
---------------------------------------------------------------------------~(*)
"""
import pytest
from surface_tracker.utils import left_rotation, right_rotation


def test_rotation():
    arr = [1, 2, 3, 4]
    n = len(arr)

    assert left_rotation(arr, 0) == [1, 2, 3, 4]
    assert left_rotation(arr, 1) == [2, 3, 4, 1]
    assert left_rotation(arr, n) == [1, 2, 3, 4]

    assert right_rotation(arr, 0) == [1, 2, 3, 4]
    assert right_rotation(arr, 1) == [4, 1, 2, 3]
    assert right_rotation(arr, n) == [1, 2, 3, 4]
