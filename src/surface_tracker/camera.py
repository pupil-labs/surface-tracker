import abc
import typing as T

import numpy as np


class CameraModel(abc.ABC):

    @property
    @abc.abstractmethod
    def resolution(self) -> T.Tuple[int, int]:
        """Returns `(width, height)`
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def distort_points_on_image_plane(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def undistort_points_on_image_plane(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
