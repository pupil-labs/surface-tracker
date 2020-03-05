import collections
import logging
import typing as T

import numpy as np

from .camera import CameraModel
from .corner import CornerId
from .heatmap import SurfaceHeatmap
from .image_crop import SurfaceImageCrop
from .marker import Marker, MarkerId
from .location import SurfaceLocation
from .surface import Surface, SurfaceId
from .visual_anchors import SurfaceVisualAnchors


logger = logging.getLogger(__name__)


class SurfaceTracker:
    def __init__(self, camera_model: CameraModel):
        self.__camera_model = camera_model
        self.__locations_tracker = _SurfaceTrackerWeakLocationStore()
        self.__argument_validator = _SurfaceTrackerArgumentValidator()

    ### Creating a surface

    def define_surface(self, name: str, markers: T.List[Marker]) -> T.Optional[Surface]:
        return Surface._create_surface_from_markers(
            name=name, markers=markers, camera_model=self.__camera_model
        )

    ### Inspecting a surface

    def surface_corner_positions_in_image_space(
        self, surface: Surface, location: SurfaceLocation, corners: T.List[CornerId]
    ) -> T.Mapping[CornerId, T.Tuple[int, int]]:
        """Return the corner positions in image space.
        """
        if len(corners) == 0:
            return {}

        positions = self.surface_points_in_image_space(
            surface=surface,
            location=location,
            points=[corner.as_tuple() for corner in corners],
            compensate_distortion=False,
        )

        assert len(corners) == len(positions)  # sanity check

        return dict(zip(corners, positions))

    def surface_points_in_image_space(
        self, surface: Surface, location: SurfaceLocation, points: T.List[T.Tuple[float, float]], compensate_distortion: bool = False,
    ) -> T.List[T.Tuple[int, int]]:
        """Transform a list of points in surface space into a list of points in image space.
        """
        if len(points) == 0:
            return []

        return location._map_from_surface_to_image(
            points=np.array(points, dtype=np.float32),
            camera_model=self.__camera_model,
            compensate_distortion=compensate_distortion,
        ).tolist()

    ### Modifying a surface

    def move_surface_corner_positions_in_image_space(
        self,
        surface: Surface,
        location: SurfaceLocation,
        new_positions: T.Mapping[CornerId, T.Tuple[int, int]],
    ):
        if len(new_positions) == 0:
            return

        ordered_corners = list(new_positions.keys())
        ordered_positions = [new_positions[corner] for corner in ordered_corners]

        ordered_position_in_surface_space_distorted = location._map_from_image_to_surface(
            points=np.array(ordered_positions, dtype=np.float32),
            camera_model=self.__camera_model,
            compensate_distortion=False,
        ).tolist()

        ordered_position_in_surface_space_undistorted = location._map_from_image_to_surface(
            points=np.array(ordered_positions, dtype=np.float32),
            camera_model=self.__camera_model,
            compensate_distortion=True,
        ).tolist()

        corner_updates = zip(ordered_corners, ordered_position_in_surface_space_distorted, ordered_position_in_surface_space_undistorted)

        for (corner, new_distorted, new_undistorted) in corner_updates:
            # TODO: Provide Surface API for moving multiple corners in one call
            surface._move_corner(
                corner=corner,
                new_position_in_surface_space_distorted=new_distorted,
                new_position_in_surface_space_undistorted=new_undistorted,
            )

    def add_markers_to_surface(
        self, surface: Surface, location: SurfaceLocation, markers: T.List[Marker], ignore_location_staleness: bool = False
    ):
        """
        """

        # Validate the surface definition and the surface location argumentse
        self.__argument_validator.validate_surface_and_location(surface=surface, location=location, ignore_location_staleness=ignore_location_staleness)

        # Since this action mutates the surface definition, mark previously computed locations as stale
        self.__locations_tracker.mark_locations_as_stale_for_surface(surface=surface)

        # TODO: Check for markers uniqueness

        # TODO: Check markers are not already part of the surface definition

        for marker in markers:
            marker_distorted = location._map_marker_from_image_to_surface(
                marker=marker,
                camera_model=self.__camera_model,
                compensate_distortion=False,
            )

            marker_undistorted = location._map_marker_from_image_to_surface(
                marker=marker,
                camera_model=self.__camera_model,
                compensate_distortion=True,
            )

            surface._add_marker(
                marker_distorted=marker_distorted,
                marker_undistorted=marker_undistorted,
            )

    def remove_marker_from_surface(
        self, surface: Surface, location: SurfaceLocation, marker_uid: MarkerId
    ):
        surface._remove_marker(marker_uid=marker_uid)

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

    def locate_surface_image_crop(
        self,
        surface: Surface,
        location: SurfaceLocation,
        width: T.Optional[int]=None,
        height: T.Optional[int]=None,
    ) -> SurfaceImageCrop:
        return SurfaceImageCrop._create_image_crop(
            location=location,
            camera_model=self.__camera_model,
            width=width,
            height=height,
        )

    def locate_surface_image_crop_with_heatmap(
        self,
        surface: Surface,
        location: SurfaceLocation,
        points: T.List[T.Tuple[int, int]],
        width: T.Optional[int]=None,
        height: T.Optional[int]=None,
    ) -> (SurfaceImageCrop, SurfaceHeatmap):

        image_crop = self.locate_surface_image_crop(
            surface=surface,
            location=location,
            width=width,
            height=height,
        )

        heatmap = SurfaceHeatmap._create_surface_heatmap(
            points_in_image_space=points,
            location=location,
            camera_model=self.__camera_model,
        )

        return (image_crop, heatmap)


##### Private Helpers


import weakref


class _SurfaceTrackerWeakLocationStore:

    def __init__(self):
        self.__storage = {}

    def track_location_for_surface(self, surface: Surface, location: SurfaceLocation):
        weak_location = weakref.ref(location)
        surface_locations = self.__storage.get(surface.uid, [])
        surface_locations.append(weak_location)
        self.__storage[surface.uid] = surface_locations

    def mark_locations_as_stale_for_surface(self, surface: Surface):
        # Get all locations that are still being used
        surface_locations = self.__storage.get(surface.uid, [])
        surface_locations = map(lambda l: l(), surface_locations)
        surface_locations = filter(lambda l: l is not None, surface_locations)

        # Invalidate all locations
        for location in surface_locations:
            location._mark_as_stale()

        # Remove locations from tracking
        self.__storage[surface.uid] = []


class _SurfaceTrackerArgumentValidator:

    def __init__(self):
        pass

    @staticmethod
    def validate_surface(surface: Surface):
        """Validate the standalone `surface` argument
        """

        if not isinstance(surface, Surface):
            raise ValueError(f"Expected an instance of Surface, but got \"{surface.__class__}\"")

    @staticmethod
    def validate_surface_and_location(surface: Surface, location: SurfaceLocation, ignore_location_staleness: bool):
        """Validate the pair of `surface` and `location` arguments
        """

        _SurfaceTrackerArgumentValidator.validate_surface(surface=surface)

        if not isinstance(location, SurfaceLocation):
            raise ValueError(f"Expected an instance of SurfaceLocation, but got \"{location.__class__}\"")

        if surface.uid != location.surface_uid:
            raise ValueError(f"SurfaceId missmatch: location doesn't belong to surface")

        if (not ignore_location_staleness) and location.is_stale:
            raise ValueError(f"Stale location: the surface definition has changed; location must be recomputed")
