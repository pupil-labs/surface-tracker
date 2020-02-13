import pytest
from surface_tracker.corner import CornerId


def test_corner_id_all_corners():

    # Test default arguments

    assert [
        CornerId.TOP_LEFT,
        CornerId.TOP_RIGHT,
        CornerId.BOTTOM_RIGHT,
        CornerId.BOTTOM_LEFT,
    ] == CornerId.all_corners()

    assert [
        CornerId.TOP_LEFT,
        CornerId.TOP_RIGHT,
        CornerId.BOTTOM_RIGHT,
        CornerId.BOTTOM_LEFT,
    ] == CornerId.all_corners(starting_with=CornerId.TOP_LEFT)

    assert [
        CornerId.TOP_LEFT,
        CornerId.TOP_RIGHT,
        CornerId.BOTTOM_RIGHT,
        CornerId.BOTTOM_LEFT,
    ] == CornerId.all_corners(clockwise=True)

    # Test different configurations

    assert [
        CornerId.TOP_LEFT,
        CornerId.TOP_RIGHT,
        CornerId.BOTTOM_RIGHT,
        CornerId.BOTTOM_LEFT,
    ] == CornerId.all_corners(starting_with=CornerId.TOP_LEFT, clockwise=True)

    assert [
        CornerId.TOP_LEFT,
        CornerId.BOTTOM_LEFT,
        CornerId.BOTTOM_RIGHT,
        CornerId.TOP_RIGHT,
    ] == CornerId.all_corners(starting_with=CornerId.TOP_LEFT, clockwise=False)

    assert [
        CornerId.BOTTOM_RIGHT,
        CornerId.BOTTOM_LEFT,
        CornerId.TOP_LEFT,
        CornerId.TOP_RIGHT,
    ] == CornerId.all_corners(starting_with=CornerId.BOTTOM_RIGHT, clockwise=True)

    assert [
        CornerId.BOTTOM_RIGHT,
        CornerId.TOP_RIGHT,
        CornerId.TOP_LEFT,
        CornerId.BOTTOM_LEFT,
    ] == CornerId.all_corners(starting_with=CornerId.BOTTOM_RIGHT, clockwise=False)


def test_corner_id_serialization():
    assert (1, 0) == CornerId.TOP_RIGHT.as_tuple()
    assert CornerId.TOP_RIGHT == CornerId.from_tuple((1, 0))
