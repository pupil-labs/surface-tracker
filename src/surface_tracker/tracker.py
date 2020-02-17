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
    def __init__(self, camera_model: CameraModel):
        self.__camera_model = camera_model

    ### Creating a surface

    def define_surface(self, name: str, markers: T.List[Marker]) -> T.Optional[Surface]:
        return Surface._create_surface_from_markers(
            name=name, markers=markers, camera_model=self.__camera_model
        )

    ### Inspecting a surface

    def surface_corner_position_in_image_space(
        self, surface: Surface, location: SurfaceLocation, corner: CornerId
    ) -> T.Tuple[int, int]:
        return location._map_from_surface_to_image(
            points=np.array([corner.as_tuple()], dtype=np.float32),
            camera_model=self.__camera_model,
            compensate_distortion=False,
        )[0].tolist()

    ### Modifying a surface

    def move_surface_corner_position_in_image_space(
        self,
        surface: Surface,
        location: SurfaceLocation,
        corner: CornerId,
        new_position: T.Tuple[int, int],
    ):

        new_position_in_surface_space_distorted = location._map_from_image_to_surface(
            points=np.array([new_position], dtype=np.float32),
            camera_model=self.__camera_model,
            compensate_distortion=False,
        )[0].tolist()

        new_position_in_surface_space_undistorted = location._map_from_image_to_surface(
            points=np.array([new_position], dtype=np.float32),
            camera_model=self.__camera_model,
            compensate_distortion=False,
        )[0].tolist()

        surface._move_corner(
            corner=corner,
            new_position_in_surface_space_distorted=new_position_in_surface_space_distorted,
            new_position_in_surface_space_undistorted=new_position_in_surface_space_undistorted,
        )

    def add_marker_to_surface(
        self, surface: Surface, location: SurfaceLocation, marker: Marker
    ):
        raise NotImplementedError()

    def remove_marker_from_surface(
        self, surface: Surface, location: SurfaceLocation, marker_uid: MarkerId
    ):
        raise NotImplementedError()

    ### Locating a surface

    def locate_surface(
        self, surface: Surface, markers: T.List[Marker]
    ) -> T.Optional[SurfaceLocation]:
        """Computes a SurfaceLocation based on a list of visible markers
        """
        return SurfaceLocation._create_location_from_markers(
            surface=surface, markers=markers, camera_model=self.__camera_model
        )

    def locate_surface_visual_anchors(
        self, surface: Surface, location: SurfaceLocation
    ) -> T.Optional[SurfaceVisualAnchors]:
        return SurfaceVisualAnchors._create_from_location(
            location=location, camera_model=self.__camera_model
        )
