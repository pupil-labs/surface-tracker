"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs
Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See LICENSE for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import typing as T

from .coordinate_space import CoordinateSpace
from .corner import CornerId

MarkerId = T.NewType("MarkerId", str)


class Marker(abc.ABC):

    # ## Abstract members

    @property
    @abc.abstractmethod
    def uid(self) -> MarkerId:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def coordinate_space(self) -> CoordinateSpace:
        raise NotImplementedError()

    def vertices(self) -> T.List[T.Tuple[float, float]]:
        # TODO: Add option to explicitly define the order of the vertices list
        # e.g.: marker.vertices(starting_with=CornerId.BOTTOM_RIGHT, clockwise=False)
        order = CornerId.all_corners()
        return self._vertices_in_order(order=order)

    @abc.abstractmethod
    def _vertices_in_order(
        self, order: T.List[CornerId]
    ) -> T.List[T.Tuple[float, float]]:
        raise NotImplementedError()

    # ## Factory

    @staticmethod
    def from_vertices(
        uid: MarkerId,
        undistorted_image_space_vertices: T.List[T.Tuple[int, int]],
        starting_with: CornerId,
        clockwise: bool,
    ) -> "Marker":
        coordinate_space = CoordinateSpace.IMAGE_UNDISTORTED
        corners = CornerId.all_corners(starting_with=starting_with, clockwise=clockwise)

        expected_len = len(corners)
        actual_len = len(undistorted_image_space_vertices)
        if expected_len != actual_len:
            raise ValueError(
                f'Expected "vertices" to have a lenght of {expected_len}, but got {actual_len}'
            )
        vertices_by_corner_id = dict(zip(corners, undistorted_image_space_vertices))

        return _Marker(
            uid=uid,
            coordinate_space=coordinate_space,
            vertices_by_corner_id=vertices_by_corner_id,
        )

    # ## Serialize

    @staticmethod
    @abc.abstractmethod
    def from_dict(value: dict) -> "Marker":
        return _Marker.from_dict(value)

    @abc.abstractmethod
    def as_dict(self) -> dict:
        raise NotImplementedError()


# #### Concrete implementations


class _Marker(Marker):
    @property
    def uid(self) -> MarkerId:
        return self.__uid

    @property
    def coordinate_space(self) -> CoordinateSpace:
        return self.__coordinate_space

    def _vertices_in_order(
        self, order: T.List[CornerId]
    ) -> T.List[T.Tuple[float, float]]:
        mapping = self.__vertices_by_corner_id
        return [mapping[c] for c in order]

    @staticmethod
    def from_dict(value: dict) -> "Marker":
        try:
            return _Marker(
                uid=value["uid"],
                coordinate_space=value["space"],
                vertices_by_corner_id=value["vertices"],
            )
        except Exception as err:
            raise ValueError(err)

    def as_dict(self) -> dict:
        return {
            "uid": self.__uid,
            "space": self.__coordinate_space,
            "vertices": self.__vertices_by_corner_id,
        }

    def __init__(
        self,
        uid: MarkerId,
        coordinate_space: CoordinateSpace,
        vertices_by_corner_id: T.Mapping[CornerId, T.Tuple[float, float]],
    ):
        self.__uid = uid
        self.__coordinate_space = coordinate_space
        self.__vertices_by_corner_id = vertices_by_corner_id
