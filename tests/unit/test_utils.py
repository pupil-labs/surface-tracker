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
