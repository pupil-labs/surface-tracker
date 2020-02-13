import abc
import enum
import typing as T

import numpy as np
import cv2

from .camera import CameraModel
from .corner import CornerId
from .marker import Marker, MarkerId, _MarkerInSurfaceSpace
from .utils import bounding_quadrangle


SurfaceId = T.NewType("SurfaceId", str)


class Surface(abc.ABC):

    ### Info
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
        marker_uids_diff = marker_uids_distorted.symmetric_difference(marker_uids_undistorted)

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
    def as_dict(self) -> dict:
        raise NotImplementedError()
