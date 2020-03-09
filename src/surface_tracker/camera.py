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
