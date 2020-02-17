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
    def _registered_markers_by_uid_distorted(self) -> T.Mapping[MarkerId, Marker]:
        raise NotImplementedError()

    @_registered_markers_by_uid_distorted.setter
    @abc.abstractmethod
    def _registered_markers_by_uid_distorted(self, value: T.Mapping[MarkerId, Marker]):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def _registered_markers_by_uid_undistorted(self) -> T.Mapping[MarkerId, Marker]:
        raise NotImplementedError()

    @_registered_markers_by_uid_undistorted.setter
    @abc.abstractmethod
    def _registered_markers_by_uid_undistorted(self, value: T.Mapping[MarkerId, Marker]):
        raise NotImplementedError()

    @property
    def registered_marker_uids(self) -> T.Set[MarkerId]:
        marker_uids_distorted = set(self._registered_markers_by_uid_distorted.keys())
        marker_uids_undistorted = set(self._registered_markers_by_uid_undistorted.keys())
        marker_uids_diff = marker_uids_distorted.symmetric_difference(
            marker_uids_undistorted
        )

        for uid in marker_uids_diff:
            logger.debug(f"Removing inconsistently registered marker with UID: {uid}")
            del self._registered_markers_by_uid_distorted[uid]
            del self._registered_markers_by_uid_undistorted[uid]

        return set(self._registered_markers_by_uid_distorted.keys())

    ### Update

    def _add_marker(self, marker_distorted: Marker, marker_undistorted: Marker):

        if not isinstance(marker_distorted, _MarkerInSurfaceSpace):
            raise ValueError(f"Expected marker_distorted in surface space")

        if not isinstance(marker_undistorted, _MarkerInSurfaceSpace):
            raise ValueError(f"Expected marker_undistorted in surface space")

        if marker_distorted.uid != marker_undistorted.uid:
            raise ValueError(f"Expected marker marker_distorted UID to match marker_undistorted UID")

        marker_uid = marker_distorted.uid
        self._registered_markers_by_uid_distorted[marker_uid] = marker_distorted
        self._registered_markers_by_uid_undistorted[marker_uid] = marker_undistorted

    def _remove_marker(self, marker_uid: MarkerId):
        self._registered_markers_by_uid_distorted[marker_uid] = None
        self._registered_markers_by_uid_undistorted[marker_uid] = None
        del self._registered_markers_by_uid_distorted[marker_uid]
        del self._registered_markers_by_uid_undistorted[marker_uid]

    def _move_corner(self, corner: CornerId, new_position_in_surface_space_distorted: T.Tuple[float, float], new_position_in_surface_space_undistorted: T.Tuple[float, float]):

        self._registered_markers_by_uid_distorted = self.__move_corner(
            corner=corner,
            registered_markers_by_uid=self._registered_markers_by_uid_distorted,
            new_position_in_surface_space=new_position_in_surface_space_distorted,
        )

        self._registered_markers_by_uid_undistorted = self.__move_corner(
            corner=corner,
            registered_markers_by_uid=self._registered_markers_by_uid_undistorted,
            new_position_in_surface_space=new_position_in_surface_space_undistorted,
        )


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
        marker_vertices_order = CornerId.all_corners()
        vertices = [m._vertices_in_image_space(order=marker_vertices_order) for m in markers]

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
                    zip(marker_vertices_order, uv_dist.tolist())
                ),
            )
            registered_markers_undistorted[marker.uid] = _MarkerInSurfaceSpace(
                uid=marker.uid,
                vertices_in_surface_space_by_corner_id=dict(
                    zip(marker_vertices_order, uv_undist.tolist())
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

    ### Private

    def __move_corner(
        self,
        corner: CornerId,
        registered_markers_by_uid: T.Mapping[MarkerId, _MarkerInSurfaceSpace],
        new_position_in_surface_space: T.Tuple[float, float],
    ):
        order = CornerId.all_corners()

        old_corners = np.array([c.as_tuple() for c in order], dtype=np.float32)
        new_corners = np.array([new_position_in_surface_space if c is corner else c.as_tuple() for c in order], dtype=np.float32)

        transform = cv2.getPerspectiveTransform(new_corners, old_corners)

        for marker_uid, marker in registered_markers_by_uid.items():
            old_vertices = np.asarray(marker._vertices_in_surface_space(order=order), dtype=np.float32)
            new_vertices = cv2.perspectiveTransform(old_vertices.reshape((-1, 1, 2)), transform).reshape((-1, 2))
            mapping = dict(zip(order, new_vertices.tolist()))
            registered_markers_by_uid[marker_uid] = _MarkerInSurfaceSpace(
                uid=marker_uid,
                vertices_in_surface_space_by_corner_id=mapping,
            )

        return registered_markers_by_uid


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
    hull_distorted = _bounding_quadrangle(vertices)
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


def _bounding_quadrangle(vertices: np.ndarray):

    # According to OpenCV implementation, cv2.convexHull only accepts arrays with
    # 32bit floats (CV_32F) or 32bit signed ints (CV_32S).
    # See: https://github.com/opencv/opencv/blob/3.4/modules/imgproc/src/convhull.cpp#L137
    # See: https://github.com/pupil-labs/pupil/issues/1544
    vertices = np.asarray(vertices, dtype=np.float32)

    hull_points = cv2.convexHull(vertices, clockwise=False)

    # The convex hull of a list of markers must have at least 4 corners, since a
    # single marker already has 4 corners. If the convex hull has more than 4
    # corners we reduce that number with approximations of the hull.
    if len(hull_points) > 4:
        new_hull = cv2.approxPolyDP(hull_points, epsilon=1, closed=True)
        if new_hull.shape[0] >= 4:
            hull_points = new_hull

    if len(hull_points) > 4:
        curvature = abs(_GetAnglesPolyline(hull_points, closed=True))
        most_acute_4_threshold = sorted(curvature)[3]
        hull_points = hull_points[curvature <= most_acute_4_threshold]

    # Vertices space is flipped in y.  We need to change the order of the
    # hull_points vertices
    hull_points = hull_points[[1, 0, 3, 2], :, :]

    # Roll the hull_points vertices until we have the right orientation:
    # vertices space has its origin at the image center. Adding 1 to the
    # coordinates puts the origin at the top left.
    distance_to_top_left = np.sqrt(
        (hull_points[:, :, 0] + 1) ** 2 + (hull_points[:, :, 1] + 1) ** 2
    )
    bot_left_idx = np.argmin(distance_to_top_left) + 1
    hull_points = np.roll(hull_points, -bot_left_idx, axis=0)
    return hull_points


# From pupil_src/shared_modules/methods.py
def _GetAnglesPolyline(polyline, closed=False):
    """
    see: http://stackoverflow.com/questions/3486172/angle-between-3-points
    ported to numpy
    returns n-2 signed angles
    """

    points = polyline[:, 0]

    if closed:
        a = np.roll(points, 1, axis=0)
        b = points
        c = np.roll(points, -1, axis=0)
    else:
        a = points[0:-2]  # all "a" points
        b = points[1:-1]  # b
        c = points[2:]  # c points
    # ab =  b.x - a.x, b.y - a.y
    ab = b - a
    # cb =  b.x - c.x, b.y - c.y
    cb = b - c
    # float dot = (ab.x * cb.x + ab.y * cb.y); # dot product
    # print 'ab:',ab
    # print 'cb:',cb

    # float dot = (ab.x * cb.x + ab.y * cb.y) dot product
    # dot  = np.dot(ab,cb.T) # this is a full matrix mulitplication we only need the diagonal \
    # dot = dot.diagonal() #  because all we look for are the dotproducts of corresponding vectors (ab[n] and cb[n])
    dot = np.sum(
        ab * cb, axis=1
    )  # or just do the dot product of the correspoing vectors in the first place!

    # float cross = (ab.x * cb.y - ab.y * cb.x) cross product
    cros = np.cross(ab, cb)

    # float alpha = atan2(cross, dot);
    alpha = np.arctan2(cros, dot)
    return alpha * (180.0 / np.pi)  # degrees
    # return alpha #radians


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
    def _registered_markers_by_uid_distorted(self) -> T.Mapping[MarkerId, Marker]:
        return self.__registered_markers_by_uid_distorted

    @_registered_markers_by_uid_distorted.setter
    def _registered_markers_by_uid_distorted(self, value: T.Mapping[MarkerId, Marker]):
        self.__registered_markers_by_uid_distorted = value

    @property
    def _registered_markers_by_uid_undistorted(self) -> T.Mapping[MarkerId, Marker]:
        return self.__registered_markers_by_uid_undistorted

    @_registered_markers_by_uid_undistorted.setter
    def _registered_markers_by_uid_undistorted(self, value: T.Mapping[MarkerId, Marker]):
        self.__registered_markers_by_uid_undistorted = value

    def as_dict(self) -> dict:
        registered_markers_distorted = self._registered_markers_by_uid_distorted
        registered_markers_distorted = dict(
            (k, v.as_dict()) for k, v in registered_markers_distorted.items()
        )

        registered_markers_undistorted = self._registered_markers_by_uid_undistorted
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
        self.__registered_markers_by_uid_distorted = registered_markers_distorted
        self.__registered_markers_by_uid_undistorted = registered_markers_undistorted
