import abc
import collections
import enum
import logging
import typing as T
import uuid

import numpy as np
import cv2

from .camera import CameraModel
from .corner import CornerId
from .marker import Marker, MarkerId, _MarkerInSurfaceSpace
from .utils import bounding_quadrangle


logger = logging.getLogger(__name__)


SurfaceId = T.NewType("SurfaceId", str)


class Surface(abc.ABC):

    __versioned_subclasses = {}

    def __init_subclass__(cls):
        storage = Surface.__versioned_subclasses
        version = cls.version
        if version is None:
            raise ValueError(
                f'Surface subclass {cls.__name__} must overwrite class property "version"'
            )
        if version in storage:
            raise ValueError(
                f"Surface subclass {cls.__name__} defines an already registered version {version}"
            )
        storage[version] = cls
        return super().__init_subclass__()

    ### Abstract members

    version = None  # type: ClassVar[int]

    @property
    @abc.abstractmethod
    def uid(self) -> SurfaceId:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def registered_markers_by_uid_distorted(self) -> T.Mapping[MarkerId, Marker]:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def registered_markers_by_uid_undistorted(self) -> T.Mapping[MarkerId, Marker]:
        raise NotImplementedError()

    @property
    def registered_marker_uids(self) -> T.Set[MarkerId]:
        marker_uids_distorted = set(self.registered_markers_by_uid_distorted.keys())
        marker_uids_undistorted = set(self.registered_markers_by_uid_undistorted.keys())
        marker_uids_diff = marker_uids_distorted.symmetric_difference(
            marker_uids_undistorted
        )

        for uid in marker_uids_diff:
            logger.debug(f"Removing inconsistently registered marker with UID: {uid}")
            del self.registered_markers_by_uid_distorted[uid]
            del self.registered_markers_by_uid_undistorted[uid]

        return set(self.registered_markers_by_uid_distorted.keys())

    ### Update

    def add_marker(self, marker: Marker):
        raise NotImplementedError()

    def remove_marker(self, marker_uid: MarkerId):
        raise NotImplementedError()

    def move_corner(self, corner_uid: CornerId, new_position):
        # TODO: Type annotate new_position
        raise NotImplementedError()

    ### Serialize

    @abc.abstractmethod
    def as_dict(self) -> dict:
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def from_dict(value: dict) -> "Surface":
        version = value["version"]
        surface_class = Surface.__versioned_subclasses.get(version, None)
        if surface_class is None:
            raise ValueError(
                f"No concrete implementation for Surface version {version}"
            )
        return surface_class.from_dict(value)

    ### Factory

    @staticmethod
    def _create_surface_from_markers(
        name: str, markers: T.List[Marker], camera_model: CameraModel
    ) -> T.Optional["Surface"]:
        if not markers:
            return None

        markers = _check_markers_uniqueness(markers)
        corners = CornerId.all_corners()
        vertices = [m._vertices_in_image_space() for m in markers]

        vertices_distorted = np.array(vertices, dtype=np.float32)
        vertices_distorted.shape = (-1, 2)
        vertices_undistorted = camera_model.undistort_points_on_image_plane(
            vertices_distorted
        )

        marker_surface_coordinates_distorted = _get_marker_surface_coordinates(
            vertices_distorted
        )
        marker_surface_coordinates_undistorted = _get_marker_surface_coordinates(
            vertices_undistorted
        )

        # Reshape to [marker, marker...]
        # where marker = [[u1, v1], [u2, v2], [u3, v3], [u4, v4]]
        marker_surface_coordinates_distorted.shape = (-1, 4, 2)
        marker_surface_coordinates_undistorted.shape = (-1, 4, 2)

        registered_markers_distorted = {}
        registered_markers_undistorted = {}

        # Add observations to library
        for marker, uv_dist, uv_undist in zip(
            markers,
            marker_surface_coordinates_distorted,
            marker_surface_coordinates_undistorted,
        ):
            registered_markers_distorted[marker.uid] = _MarkerInSurfaceSpace(
                uid=marker.uid,
                vertices_in_surface_space_by_corner_id=dict(
                    zip(corners, uv_dist.tolist())
                ),
            )
            registered_markers_undistorted[marker.uid] = _MarkerInSurfaceSpace(
                uid=marker.uid,
                vertices_in_surface_space_by_corner_id=dict(
                    zip(corners, uv_undist.tolist())
                ),
            )

        # Create a new unique identifier for this surface
        uid = str(uuid.uuid4())
        uid = SurfaceId(uid)

        return _Surface_V2(
            uid=uid,
            name=name,
            registered_markers_distorted=registered_markers_distorted,
            registered_markers_undistorted=registered_markers_undistorted,
        )


def _check_markers_uniqueness(markers: T.List[Marker]) -> T.List[Marker]:
    non_unique_markers_by_uid = collections.OrderedDict()

    for m in markers:
        non_unique = non_unique_markers_by_uid.get(m.uid, [])
        non_unique.append(m)
        non_unique_markers_by_uid[m.uid] = non_unique

    unique_markers = []

    for uid, non_unique in non_unique_markers_by_uid.items():
        if len(non_unique) > 1:
            logger.debug(f"Duplicate markers with uid {uid}.")
        unique_markers.append(non_unique[0])

    return unique_markers


def _get_marker_surface_coordinates(vertices: np.ndarray) -> np.ndarray:
    hull_distorted = bounding_quadrangle(vertices)
    transform_candidate_image_to_surface = _get_transform_to_normalized_corners(
        hull_distorted
    )
    shape = vertices.shape
    vertices.shape = (-1, 1, 2)
    transform = cv2.perspectiveTransform(vertices, transform_candidate_image_to_surface)
    vertices.shape = shape
    return transform


def _get_transform_to_normalized_corners(vertices: np.ndarray) -> np.ndarray:
    corners = CornerId.all_corners(starting_with=CornerId.TOP_LEFT, clockwise=True)
    corners = [c.as_tuple() for c in corners]
    corners = np.array(corners, dtype=np.float32)
    return cv2.getPerspectiveTransform(vertices, corners)


##### Concrete implementations


class _Surface_V2(Surface):

    version = 2

    @property
    def uid(self) -> SurfaceId:
        return self.__uid

    @property
    def name(self) -> str:
        return self.__name

    @property
    def registered_markers_by_uid_distorted(self) -> T.Mapping[MarkerId, Marker]:
        return self.__registered_markers_distorted

    @property
    def registered_markers_by_uid_undistorted(self) -> T.Mapping[MarkerId, Marker]:
        return self.__registered_markers_undistorted

    def as_dict(self) -> dict:
        registered_markers_distorted = self.__registered_markers_distorted
        registered_markers_distorted = dict(
            (k, v.as_dict()) for k, v in registered_markers_distorted.items()
        )

        registered_markers_undistorted = self.__registered_markers_undistorted
        registered_markers_undistorted = dict(
            (k, v.as_dict()) for k, v in registered_markers_undistorted.items()
        )

        return {
            "version": self.version,
            "uid": str(self.uid),
            "name": self.name,
            "registered_markers_distorted": registered_markers_distorted,
            "registered_markers_undistorted": registered_markers_undistorted,
        }

    @staticmethod
    def from_dict(value: dict) -> "Surface":
        try:
            actual_version = value["version"]
            expected_version = _Surface_V2.version
            assert (
                expected_version == actual_version
            ), f"Surface version missmatch; expected {expected_version}, but got {actual_version}"

            registered_markers_distorted = version["registered_markers_distorted"]
            registered_markers_distorted = dict(
                (k, Marker.from_dict(v))
                for k, v in registered_markers_distorted.items()
            )

            registered_markers_undistorted = version["registered_markers_undistorted"]
            registered_markers_undistorted = dict(
                (k, Marker.from_dict(v))
                for k, v in registered_markers_undistorted.items()
            )

            return _Surface_V2(
                uid=SurfaceId(value["uid"]),
                name=value["name"],
                registered_markers_distorted=registered_markers_distorted,
                registered_markers_undistorted=registered_markers_undistorted,
            )
        except Exception as err:
            raise ValueError(err)

    def __init__(
        self,
        uid: SurfaceId,
        name: str,
        registered_markers_distorted: T.Mapping[MarkerId, _MarkerInSurfaceSpace],
        registered_markers_undistorted: T.Mapping[MarkerId, _MarkerInSurfaceSpace],
    ):
        self.__uid = uid
        self.__name = name
        self.__registered_markers_distorted = registered_markers_distorted
        self.__registered_markers_undistorted = registered_markers_undistorted
