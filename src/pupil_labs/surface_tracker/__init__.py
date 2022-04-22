"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs
Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See LICENSE for license details.
---------------------------------------------------------------------------~(*)
"""
from . import utils
from .camera import Camera
from .coordinate_space import CoordinateSpace
from .corner import CornerId
from .heatmap import SurfaceHeatmap
from .image_crop import SurfaceImageCrop
from .location import SurfaceLocation
from .marker import Marker, MarkerId
from .orientation import SurfaceOrientation
from .surface import Surface, SurfaceId
from .tracker import SurfaceTracker
from .visual_anchors import SurfaceVisualAnchors

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("pupil_labs.project_name")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

__all__ = [
    "__version__",
    "utils",
    "Camera",
    "CoordinateSpace",
    "CornerId",
    "SurfaceHeatmap",
    "SurfaceImageCrop",
    "SurfaceLocation",
    "Marker",
    "MarkerId",
    "SurfaceOrientation",
    "Surface",
    "SurfaceId",
    "SurfaceTracker",
    "SurfaceVisualAnchors",
]
