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
import logging
import typing as T

import cv2
import numpy as np

from .coordinate_space import CoordinateSpace
from .corner import CornerId
from .marker import Marker, _Marker
from .surface import Surface, SurfaceId

logger = logging.getLogger(__name__)


class SurfaceLocation(abc.ABC):

    __versioned_subclasses = {}

    def __init_subclass__(cls):
        storage = SurfaceLocation.__versioned_subclasses
        version = cls.version
        if version is None:
            raise ValueError(
                f'SurfaceLocation subclass {cls.__name__} must overwrite class property "version"'
            )
        if version in storage:
            raise ValueError(
                f"SurfaceLocation subclass {cls.__name__} defines an already registered version {version}"
            )
        storage[version] = cls
        return super().__init_subclass__()

    # ## Abstract members

    version = None  # type: ClassVar[int]

    @property
    @abc.abstractmethod
    def surface_uid(self) -> SurfaceId:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def is_stale(self) -> bool:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def number_of_markers_detected(self) -> int:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def transform_matrix_from_image_to_surface_undistorted(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def transform_matrix_from_surface_to_image_undistorted(self) -> np.ndarray:
        raise NotImplementedError()

    # ## Factory

    @staticmethod
    def _create_location_from_markers(
        surface: Surface, markers: T.List[Marker]
    ) -> "SurfaceLocation":

        if surface is None:
            raise ValueError("Surface is None")

        if not all(
            m.coordinate_space == CoordinateSpace.IMAGE_UNDISTORTED for m in markers
        ):
            raise ValueError("Expected all markers to be in undistorted image space")

        registered_marker_uids = set(surface.registered_marker_uids)

        registered_markers_by_uid_undistorted = (
            surface._registered_markers_by_uid_undistorted
        )

        # Return None for an invalid surface definition
        if len(registered_marker_uids) == 0:
            logger.debug(f"Surface does not have any registered markers that define it")
            return None

        # Get the marker UIDs for the visible markers
        visible_markers_by_uid = {m.uid: m for m in markers}
        if len(visible_markers_by_uid) < len(markers):
            raise ValueError(f"Detected markers must be unique")

        # Only keep the visible marker UIDs for markers that define the surface
        visible_markers_by_uid = {
            uid: m
            for uid, m in visible_markers_by_uid.items()
            if uid in registered_marker_uids
        }

        # If no markers are visible - return
        if len(visible_markers_by_uid) == 0:
            return None

        # Get the set of marker UIDs that are both visible and registered
        matching_marker_uids = set(visible_markers_by_uid.keys())

        marker_vertices_order = CornerId.all_corners()

        visible_vertices_undistorted = np.array(
            [
                visible_markers_by_uid[uid]._vertices_in_order(
                    order=marker_vertices_order
                )
                for uid in matching_marker_uids
            ]
        )
        visible_vertices_undistorted.shape = (-1, 2)

        registered_vertices_undistorted = np.array(
            [
                registered_markers_by_uid_undistorted[uid]._vertices_in_order(
                    order=marker_vertices_order
                )
                for uid in matching_marker_uids
            ]
        )
        registered_vertices_undistorted.shape = (-1, 2)

        (
            transform_matrix_from_image_to_surface_undistorted,
            transform_matrix_from_surface_to_image_undistorted,
        ) = _find_homographies(
            registered_vertices_undistorted, visible_vertices_undistorted
        )

        return _SurfaceLocation_v2(
            surface_uid=surface.uid,
            number_of_markers_detected=len(matching_marker_uids),
            transform_matrix_from_image_to_surface_undistorted=transform_matrix_from_image_to_surface_undistorted,
            transform_matrix_from_surface_to_image_undistorted=transform_matrix_from_surface_to_image_undistorted,
        )

    # ## Mapping

    def _map_from_image_to_surface(
        self, points: np.ndarray, transform_matrix=None
    ) -> np.ndarray:
        return self.__map_points(
            points=points,
            transform_matrix=self.__image_to_surface_transform(
                transform_matrix=transform_matrix
            ),
            custom_transformation=False,
        )

    def _map_from_surface_to_image(
        self, points: np.ndarray, transform_matrix=None
    ) -> np.ndarray:
        return self.__map_points(
            points=points,
            transform_matrix=self.__surface_to_image_transform(
                transform_matrix=transform_matrix
            ),
            custom_transformation=True,
        )

    def _map_marker_from_image_to_surface(
        self, marker: Marker, transform_matrix=None
    ) -> Marker:

        order = CornerId.all_corners()

        vertices_in_image_space = marker._vertices_in_order(order=order)

        vertices_in_image_space_numpy = np.asarray(vertices_in_image_space).reshape(
            (4, 2)
        )
        vertices_in_surface_space_numpy = self._map_from_image_to_surface(
            points=vertices_in_image_space_numpy, transform_matrix=transform_matrix
        )

        vertices_in_surface_space = vertices_in_surface_space_numpy.tolist()

        assert len(order) == len(vertices_in_surface_space), "Sanity check"
        vertices_by_corner_id = dict(zip(order, vertices_in_surface_space))

        return _Marker(
            uid=marker.uid,
            coordinate_space=CoordinateSpace.SURFACE_UNDISTORTED,
            vertices_by_corner_id=vertices_by_corner_id,
        )

    def _map_marker_from_surface_to_image(
        self, marker: Marker, transform_matrix=None
    ) -> Marker:

        order = CornerId.all_corners()

        vertices_in_surface_space = marker._vertices_in_order(order=order)

        vertices_in_surface_space_numpy = np.asarray(vertices_in_surface_space).reshape(
            (4, 2)
        )
        vertices_in_image_space_numpy = self._map_from_surface_to_image(
            points=vertices_in_surface_space_numpy, transform_matrix=transform_matrix
        )

        vertices_in_image_space = vertices_in_image_space_numpy.tolist()

        assert len(order) == len(vertices_in_image_space), "Sanity check"
        vertices_by_corner_id = dict(zip(order, vertices_in_image_space))

        return _Marker(
            uid=marker.uid,
            coordinate_space=CoordinateSpace.IMAGE_UNDISTORTED,
            vertices_by_corner_id=vertices_by_corner_id,
        )

    # ## Serialize

    @abc.abstractmethod
    def as_dict(self) -> dict:
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def from_dict(value: dict) -> "SurfaceLocation":
        version = value["version"]
        location_class = SurfaceLocation.__versioned_subclasses.get(version, None)
        if location_class is None:
            raise ValueError(
                f"No concrete implementation for SurfaceLocation version {version}"
            )
        return location_class.from_dict(value)

    # ## Private

    def __image_to_surface_transform(
        self, transform_matrix: T.Optional[np.ndarray]
    ) -> np.ndarray:
        if transform_matrix is not None:
            return transform_matrix
        return self.transform_matrix_from_image_to_surface_undistorted

    def __surface_to_image_transform(
        self, transform_matrix: T.Optional[np.ndarray]
    ) -> np.ndarray:
        if transform_matrix is not None:
            return transform_matrix
        return self.transform_matrix_from_surface_to_image_undistorted

    @staticmethod
    def __map_points(
        points: np.ndarray, transform_matrix: np.ndarray, custom_transformation=False
    ) -> np.ndarray:

        points = np.asarray(points)

        # Validate points

        if len(points.shape) == 1 and points.shape[0] == 2:
            pass
        elif len(points.shape) == 2 and points.shape[1] == 2:
            pass
        else:
            raise ValueError(
                f"Expected points to have shape (2,) or (N, 2), but got {points.shape}"
            )

        # Perspective transform

        shape = points.shape
        points.shape = (-1, 1, 2)
        if custom_transformation:
            points = _perspective_transform(points, transform_matrix)
        else:
            points = cv2.perspectiveTransform(points, transform_matrix)
        points.shape = shape

        return points


# #### Helper functions


def is_clockwise_triangle(points):
    p1, p2, p3 = points
    val = (float(p2[0][1] - p1[0][1]) * (p3[0][0] - p2[0][0])) - (
        float(p2[0][0] - p1[0][0]) * (p3[0][1] - p2[0][1])
    )
    return val > 0


def orientation_quadrangle(points, clockwise=True):
    p0, p1, p2, p3 = points
    return all(
        (
            is_clockwise_triangle((p0, p1, p2)) == clockwise,
            is_clockwise_triangle((p1, p2, p3)) == clockwise,
            is_clockwise_triangle((p2, p3, p0)) == clockwise,
        )
    )


def project_points_pos_z(homogeneous_points, transform_matrix):
    res = []
    for p in homogeneous_points:
        projected_point = np.dot(transform_matrix, p[0])
        if projected_point[2] < 0:
            projected_point[2] = (
                np.linalg.norm([projected_point[0], projected_point[1]]) / 32
            )
        res.append(projected_point)
    return res


def _perspective_transform(points, transform_matrix):
    homogeneous_points = cv2.convertPointsToHomogeneous(points)

    proj_unflipped = project_points_pos_z(homogeneous_points, transform_matrix)
    points_unflipped = cv2.convertPointsFromHomogeneous(np.asarray(proj_unflipped))
    orientation_unflipped = orientation_quadrangle(points_unflipped, clockwise=True)

    proj_flipped = project_points_pos_z(-homogeneous_points, transform_matrix)
    points_flipped = cv2.convertPointsFromHomogeneous(np.asarray(proj_flipped))
    orientation_flipped = orientation_quadrangle(points_flipped, clockwise=True)

    if orientation_unflipped and not orientation_flipped:
        return points_unflipped
    if orientation_flipped and not orientation_unflipped:
        return points_flipped
    if orientation_unflipped and orientation_flipped:

        def min_dist(points):
            return min(np.linalg.norm(p) for p in points)

        if min_dist(points_flipped) < min_dist(points_unflipped):
            return points_flipped
        else:
            return points_unflipped
    return points_unflipped


def _find_homographies(points_A, points_B):
    points_A = points_A.reshape((-1, 1, 2))
    points_B = points_B.reshape((-1, 1, 2))

    B_to_A, mask = cv2.findHomography(points_A, points_B)
    # NOTE: cv2.findHomography(A, B) will not produce the inverse of
    # cv2.findHomography(B, A)! The errors can actually be quite large, resulting in
    # on-screen discrepancies of up to 50 pixels. We try to find the inverse
    # analytically instead with fallbacks.

    try:
        A_to_B = np.linalg.inv(B_to_A)
        return A_to_B, B_to_A
    except np.linalg.LinAlgError:
        logger.debug(
            "Failed to calculate inverse homography with np.inv()! "
            "Trying with np.pinv() instead."
        )

    try:
        A_to_B = np.linalg.pinv(B_to_A)
        return A_to_B, B_to_A
    except np.linalg.LinAlgError:
        logger.warning(
            "Failed to calculate inverse homography with np.pinv()! "
            "Falling back to inaccurate manual computation!"
        )

    A_to_B, mask = cv2.findHomography(points_B, points_A)
    return A_to_B, B_to_A


# #### Concrete implementations


class _SurfaceLocation_v2(SurfaceLocation):

    version = 2  # type: ClassVar[int]

    @property
    def surface_uid(self) -> SurfaceId:
        return self.__surface_uid

    @property
    def is_stale(self) -> bool:
        return self.__is_stale

    @property
    def number_of_markers_detected(self) -> int:
        return self.__number_of_markers_detected

    @property
    def transform_matrix_from_image_to_surface_undistorted(self) -> np.ndarray:
        return self.__transform_matrix_from_image_to_surface_undistorted

    @property
    def transform_matrix_from_surface_to_image_undistorted(self) -> np.ndarray:
        return self.__transform_matrix_from_surface_to_image_undistorted

    def __init__(
        self,
        surface_uid: SurfaceId,
        number_of_markers_detected: int,
        transform_matrix_from_image_to_surface_undistorted: np.ndarray,
        transform_matrix_from_surface_to_image_undistorted: np.ndarray,
    ):
        self.__is_stale = False
        self.__surface_uid = surface_uid
        self.__number_of_markers_detected = number_of_markers_detected
        self.__transform_matrix_from_image_to_surface_undistorted = (
            transform_matrix_from_image_to_surface_undistorted
        )
        self.__transform_matrix_from_surface_to_image_undistorted = (
            transform_matrix_from_surface_to_image_undistorted
        )

    def _mark_as_stale(self):
        self.__is_stale = False

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "surface_uid": str(self.surface_uid),
            "number_of_markers_detected": self.number_of_markers_detected,
            "transform_matrix_from_image_to_surface_undistorted": self.transform_matrix_from_image_to_surface_undistorted.tolist(),
            "transform_matrix_from_surface_to_image_undistorted": self.transform_matrix_from_surface_to_image_undistorted.tolist(),
        }

    @staticmethod
    def from_dict(value: dict) -> "SurfaceLocation":
        try:
            actual_version = value["version"]
            expected_version = _SurfaceLocation_v2.version
            assert (
                expected_version == actual_version
            ), f"SurfaceLocation version missmatch; expected {expected_version}, but got {actual_version}"
            return _SurfaceLocation_v2(
                surface_uid=SurfaceId(value["surface_uid"]),
                number_of_markers_detected=int(value["number_of_markers_detected"]),
                transform_matrix_from_image_to_surface_undistorted=np.asarray(
                    value["transform_matrix_from_image_to_surface_undistorted"]
                ),
                transform_matrix_from_surface_to_image_undistorted=np.asarray(
                    value["transform_matrix_from_surface_to_image_undistorted"]
                ),
            )
        except Exception as err:
            raise ValueError(err)
