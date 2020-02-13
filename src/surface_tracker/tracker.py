import collections
import logging
import typing as T

import numpy as np

from .camera import CameraModel
from .corner import CornerId
from .marker import Marker, MarkerId
from .location import SurfaceLocation
from .surface import Surface, SurfaceId
from .visual_anchors import SurfaceVisualAnchors


logger = logging.getLogger(__name__)


class SurfaceTracker:

    def __init__(self):
        pass

    def define_surface(
        self, name: str, markers: T.List[Marker], camera_model: CameraModel
    ) -> T.Optional[Surface]:
        return Surface._create_surface_from_markers(name=name, markers=markers, camera_model=camera_model)

    def locate_surface(
        self, surface: Surface, markers: T.List[Marker], camera_model: CameraModel
    ) -> T.Optional[SurfaceLocation]:
        """Computes a SurfaceLocation based on a list of visible markers
        """
        return SurfaceLocation._create_location_from_markers(
            surface=surface,
            markers=markers,
            camera_model=camera_model,
        )
