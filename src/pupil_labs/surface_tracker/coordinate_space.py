"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs
Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See LICENSE for license details.
---------------------------------------------------------------------------~(*)
"""
import enum


class CoordinateSpace(enum.Enum):
    IMAGE_DISTORTED = "image-distorted"
    IMAGE_UNDISTORTED = "image-undistorted"
    SURFACE_DISTORTED = "surface-distorted"
    SURFACE_UNDISTORTED = "surface-undisitorted"
