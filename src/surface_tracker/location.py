import abc
import typing as T

import numpy as np
import cv2

from .camera import CameraModel
from .surface import Surface, SurfaceId


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

    ### Abstract members

    version = None  # type: ClassVar[int]

    @property
    @abc.abstractmethod
    def surface_uid(self) -> SurfaceId:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def number_of_markers_detected(self) -> int:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def transform_matrix_from_image_to_surface_distorted(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def transform_matrix_from_image_to_surface_undistorted(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def transform_matrix_from_surface_to_image_distorted(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def transform_matrix_from_surface_to_image_undistorted(self) -> np.ndarray:
        raise NotImplementedError()

    ### Factory

    @staticmethod
    def create_location(
        surface_uid: SurfaceId,
        number_of_markers_detected: int,
        transform_matrix_from_image_to_surface_distorted: np.ndarray,
        transform_matrix_from_surface_to_image_distorted: np.ndarray,
        transform_matrix_from_image_to_surface_undistorted: np.ndarray,
        transform_matrix_from_surface_to_image_undistorted: np.ndarray,
    ) -> "SurfaceLocation":
        return _SurfaceLocation_v2(
            surface_uid=surface_uid,
            number_of_markers_detected=number_of_markers_detected,
            transform_matrix_from_image_to_surface_distorted=transform_matrix_from_image_to_surface_distorted,
            transform_matrix_from_surface_to_image_distorted=transform_matrix_from_surface_to_image_distorted,
            transform_matrix_from_image_to_surface_undistorted=transform_matrix_from_image_to_surface_undistorted,
            transform_matrix_from_surface_to_image_undistorted=transform_matrix_from_surface_to_image_undistorted,
        )

    ### Mapping

    def map_from_image_to_surface(
        self,
        points: np.ndarray,
        camera_model: CameraModel,
        compensate_distortion: bool = True,
        transform_matrix=None,
    ) -> np.ndarray:
        return self.__map_points(
            points=points,
            compensate_distortion=compensate_distortion,
            compensate_distortion_fn=camera_model.undistort_points_on_image_plane,
            transform_matrix=self.__image_to_surface_transform(
                compensate_distortion=compensate_distortion,
                transform_matrix=transform_matrix,
            ),
        )

    def map_from_surface_to_image(
        self,
        points: np.ndarray,
        camera_model: CameraModel,
        compensate_distortion: bool = True,
        transform_matrix=None,
    ) -> np.ndarray:
        return self.__map_points(
            points=points,
            compensate_distortion=compensate_distortion,
            compensate_distortion_fn=camera_model.distort_points_on_image_plane,
            transform_matrix=self.__surface_to_image_transform(
                compensate_distortion=compensate_distortion,
                transform_matrix=transform_matrix,
            ),
        )

    ### Serialize

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

    ### Private

    def __image_to_surface_transform(
        self, compensate_distortion: bool, transform_matrix: T.Optional[np.ndarray]
    ) -> np.ndarray:
        if transform_matrix is not None:
            return transform_matrix
        elif compensate_distortion:
            return self.transform_matrix_from_image_to_surface_undistorted
        else:
            return self.transform_matrix_from_image_to_surface_distorted

    def __surface_to_image_transform(
        self, compensate_distortion: bool, transform_matrix: T.Optional[np.ndarray]
    ) -> np.ndarray:
        if transform_matrix is not None:
            return transform_matrix
        elif compensate_distortion:
            return self.transform_matrix_from_surface_to_image_undistorted
        else:
            return self.transform_matrix_from_surface_to_image_distorted

    @staticmethod
    def __map_points(
        points: np.ndarray,
        compensate_distortion_fn,
        compensate_distortion: bool,
        transform_matrix: np.ndarray,
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

        # Distortion compensation

        if compensate_distortion:
            shape = points.shape
            points = compensate_distortion_fn(points)
            points.shape = shape

        # Perspective transform

        shape = points.shape
        points.shape = (-1, 1, 2)
        points = cv2.perspectiveTransform(points, transform_matrix)
        points.shape = shape

        return points


##### Concrete implementations


class _SurfaceLocation_v2(SurfaceLocation):

    version = 2  # type: ClassVar[int]

    @property
    def surface_uid(self) -> SurfaceId:
        return self.__surface_uid

    @property
    def number_of_markers_detected(self) -> int:
        return self.__number_of_markers_detected

    @property
    def transform_matrix_from_image_to_surface_distorted(self) -> np.ndarray:
        return self.__transform_matrix_from_image_to_surface_distorted

    @property
    def transform_matrix_from_image_to_surface_undistorted(self) -> np.ndarray:
        return self.__transform_matrix_from_image_to_surface_undistorted

    @property
    def transform_matrix_from_surface_to_image_distorted(self) -> np.ndarray:
        return self.__transform_matrix_from_surface_to_image_distorted

    @property
    def transform_matrix_from_surface_to_image_undistorted(self) -> np.ndarray:
        return self.__transform_matrix_from_surface_to_image_undistorted

    def __init__(
        self,
        surface_uid: SurfaceId,
        number_of_markers_detected: int,
        transform_matrix_from_image_to_surface_distorted: np.ndarray,
        transform_matrix_from_surface_to_image_distorted: np.ndarray,
        transform_matrix_from_image_to_surface_undistorted: np.ndarray,
        transform_matrix_from_surface_to_image_undistorted: np.ndarray,
    ):
        self.__surface_uid = surface_uid
        self.__number_of_markers_detected = number_of_markers_detected
        self.__transform_matrix_from_image_to_surface_distorted = (
            transform_matrix_from_image_to_surface_distorted
        )
        self.__transform_matrix_from_surface_to_image_distorted = (
            transform_matrix_from_surface_to_image_distorted
        )
        self.__transform_matrix_from_image_to_surface_undistorted = (
            transform_matrix_from_image_to_surface_undistorted
        )
        self.__transform_matrix_from_surface_to_image_undistorted = (
            transform_matrix_from_surface_to_image_undistorted
        )

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "surface_uid": str(self.surface_uid),
            "number_of_markers_detected": self.number_of_markers_detected,
            "transform_matrix_from_image_to_surface_distorted": self.transform_matrix_from_image_to_surface_distorted.tolist(),
            "transform_matrix_from_surface_to_image_distorted": self.transform_matrix_from_surface_to_image_distorted.tolist(),
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
                transform_matrix_from_image_to_surface_distorted=np.asarray(
                    value["transform_matrix_from_image_to_surface_distorted"]
                ),
                transform_matrix_from_surface_to_image_distorted=np.asarray(
                    value["transform_matrix_from_surface_to_image_distorted"]
                ),
                transform_matrix_from_image_to_surface_undistorted=np.asarray(
                    value["transform_matrix_from_image_to_surface_undistorted"]
                ),
                transform_matrix_from_surface_to_image_undistorted=np.asarray(
                    value["transform_matrix_from_surface_to_image_undistorted"]
                ),
            )
        except Exception as err:
            raise ValueError(err)
