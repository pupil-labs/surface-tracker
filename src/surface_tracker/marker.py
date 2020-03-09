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
import collections
import typing as T

from .corner import CornerId


MarkerId = T.NewType("MarkerId", str)


VertexInImageSpace = T.Tuple[int, int]


class Marker(abc.ABC):

    ### Abstract members

    @property
    @abc.abstractmethod
    def uid(self) -> MarkerId:
        raise NotImplementedError()

    def vertices(self) -> T.List[tuple]:
        # TODO: Add option to explicitly define the order of the vertices list
        # e.g.: marker.vertices(starting_with=CornerId.BOTTOM_RIGHT, clockwise=False)
        order = CornerId.all_corners()
        return self._vertices_in_order(order=order)

    @abc.abstractmethod
    def _vertices_in_order(self, order: T.List[CornerId]) -> T.List[tuple]:
        raise NotImplementedError()

    ### Factory

    @staticmethod
    def from_vertices(
        uid: MarkerId,
        vertices: T.List[VertexInImageSpace],
        starting_with: CornerId,
        clockwise: bool,
    ) -> "Marker":
        corners = CornerId.all_corners(starting_with=starting_with, clockwise=clockwise)

        expected_len = len(corners)
        actual_len = len(vertices)
        if expected_len != actual_len:
            raise ValueError(
                f'Expected "vertices" to have a lenght of {expected_len}, but got {actual_len}'
            )
        mapping = dict(zip(corners, vertices))

        return _MarkerInImageSpace(
            uid=uid, vertices_in_image_space_by_corner_id=mapping
        )

    ### Serialize

    @staticmethod
    @abc.abstractmethod
    def from_dict(value: dict) -> "Marker":
        try:
            coordinate_space = value["space"]
            if coordinate_space == _MarkerInImageSpace._COORDINATE_SPACE:
                return _MarkerInImageSpace.from_dict(value)
            if coordinate_space == _MarkerInSurfaceSpace._COORDINATE_SPACE:
                return _MarkerInSurfaceSpace.from_dict(value)
            raise ValueError(
                f'Vertices in unknown coordinate space "{coordinate_space}"'
            )
        except Exception as err:
            raise ValueError(err)

    @abc.abstractmethod
    def as_dict(self) -> dict:
        raise NotImplementedError()

    ### Private

    @staticmethod
    def __validate_vertices(
        vertices_in_image_space_by_corner_id: T.Mapping[CornerId, T.Tuple[int, int]]
    ):
        expected_corners = set(CornerId.all_corners())
        actual_corners = set(vertices_in_image_space_by_corner_id.keys())
        missing_corners = expected_corners.difference(actual_corners)

        if missing_corners:
            raise ValueError(f"Missing values for corners: {missing_corners}")


##### Concrete implementations


class _MarkerInImageSpace(Marker):
    @property
    def uid(self) -> MarkerId:
        return self.__uid

    def _vertices_in_order(self, order: T.List[CornerId]) -> T.List[tuple]:
        return self._vertices_in_image_space(order=order)

    @staticmethod
    def from_dict(value: dict) -> "Marker":
        try:
            actual_space = value["space"]
            expected_space = _MarkerInImageSpace._COORDINATE_SPACE
            if actual_space != expected_space:
                raise ValueError(
                    f'Missmatched coordinate space; expected "{expected_space}", but "{actual_space}"'
                )
            return _MarkerInImageSpace(
                uid=value["uid"], vertices_in_image_space_by_corner_id=value["vertices"]
            )
        except Exception as err:
            raise ValueError(err)

    def as_dict(self) -> dict:
        return {
            "uid": self.__uid,
            "space": self._COORDINATE_SPACE,
            "vertices": self.__vertices_in_image_space_by_corner_id,
        }

    ### Internal

    _COORDINATE_SPACE = "image"

    def __init__(
        self,
        uid: MarkerId,
        vertices_in_image_space_by_corner_id: T.Mapping[CornerId, T.Tuple[int, int]],
    ):
        self.__uid = uid
        self.__vertices_in_image_space_by_corner_id = (
            vertices_in_image_space_by_corner_id
        )

    def _vertices_in_image_space(
        self, order: T.List[CornerId]
    ) -> T.List[T.Tuple[int, int]]:
        mapping = self.__vertices_in_image_space_by_corner_id
        return [mapping[c] for c in order]


class _MarkerInSurfaceSpace(Marker):
    @property
    def uid(self) -> MarkerId:
        return self.__uid

    def _vertices_in_order(self, order: T.List[CornerId]) -> T.List[tuple]:
        return self._vertices_in_surface_space(order=order)

    @staticmethod
    def from_dict(value: dict) -> "Marker":
        try:
            actual_space = value["space"]
            expected_space = _MarkerInSurfaceSpace._COORDINATE_SPACE
            if actual_space != expected_space:
                raise ValueError(
                    f'Missmatched coordinate space; expected "{expected_space}", but "{actual_space}"'
                )
            return _MarkerInSurfaceSpace(
                uid=value["uid"],
                vertices_in_surface_space_by_corner_id=value["vertices"],
            )
        except Exception as err:
            raise ValueError(err)

    def as_dict(self) -> dict:
        return {
            "uid": self.__uid,
            "space": self._COORDINATE_SPACE,
            "vertices": self.__vertices_in_surface_space_by_corner_id,
        }

    ### Internal

    _COORDINATE_SPACE = "surface"

    def __init__(
        self,
        uid: MarkerId,
        vertices_in_surface_space_by_corner_id: T.Mapping[
            CornerId, T.Tuple[float, float]
        ],
    ):
        self.__uid = uid
        self.__vertices_in_surface_space_by_corner_id = (
            vertices_in_surface_space_by_corner_id
        )

    def _vertices_in_surface_space(
        self, order: T.List[CornerId]
    ) -> T.List[T.Tuple[float, float]]:
        mapping = self.__vertices_in_surface_space_by_corner_id
        return [mapping[c] for c in order]
