"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs
Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See LICENSE for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import typing as T

import numpy as np

from .camera import Camera
from .corner import CornerId
from .heatmap import SurfaceHeatmap
from .image_crop import SurfaceImageCrop
from .location import SurfaceLocation
from .marker import Marker, MarkerId
from .orientation import SurfaceOrientation
from .surface import Surface
from .visual_anchors import SurfaceVisualAnchors

logger = logging.getLogger(__name__)


class SurfaceTracker:
    def __init__(self):
        self.__locations_tracker = _SurfaceTrackerWeakLocationStore()
        self.__argument_validator = _SurfaceTrackerArgumentValidator()

    # ## Creating a surface

    def define_surface(self, name: str, markers: T.List[Marker]) -> T.Optional[Surface]:
        """ """

        return Surface._create_surface_from_markers(name=name, markers=markers)

    # ## Inspecting a surface

    def surface_corner_positions_in_image_space(
        self, surface: Surface, location: SurfaceLocation, corners: T.List[CornerId]
    ) -> T.Mapping[CornerId, T.Tuple[int, int]]:
        """Return the corner positions in image space."""

        # Validate the surface definition and the surface location arguments
        self.__argument_validator.validate_surface_and_location(
            surface=surface, location=location, ignore_location_staleness=False
        )

        if len(corners) == 0:
            return {}

        positions = self.surface_points_in_image_space(
            surface=surface,
            location=location,
            points=[corner.as_tuple() for corner in corners],
        )

        assert len(corners) == len(positions)  # sanity check

        updated_corners = surface.orientation.map_surface_corners_to_rotation(
            dict(zip(corners, positions))
        )

        return updated_corners

    def surface_points_in_image_space(
        self,
        surface: Surface,
        location: SurfaceLocation,
        points: T.List[T.Tuple[float, float]],
    ) -> T.List[T.Tuple[int, int]]:
        """Transform a list of points in surface space into a list of points in image space."""

        # Validate the surface definition and the surface location arguments
        self.__argument_validator.validate_surface_and_location(
            surface=surface, location=location, ignore_location_staleness=False
        )

        if len(points) == 0:
            return []

        return location._map_from_surface_to_image(
            points=np.array(points, dtype=np.float32)
        ).tolist()

    # ## Modifying a surface

    def move_surface_corner_positions_in_image_space(
        self,
        surface: Surface,
        location: SurfaceLocation,
        new_positions: T.Mapping[CornerId, T.Tuple[int, int]],
        ignore_location_staleness: bool = False,
    ):
        """ """

        # Validate the surface definition and the surface location argumentse
        self.__argument_validator.validate_surface_and_location(
            surface=surface,
            location=location,
            ignore_location_staleness=ignore_location_staleness,
        )
        new_positions = surface.orientation.map_surface_corners_to_default(
            new_positions
        )

        if len(new_positions) == 0:
            return

        # Since this action mutates the surface definition, mark previously computed
        # locations as stale
        self.__locations_tracker.mark_locations_as_stale_for_surface(surface=surface)

        ordered_corners = list(new_positions.keys())
        ordered_positions = [new_positions[corner] for corner in ordered_corners]

        ordered_position_in_surface_space_undistorted = (
            location._map_from_image_to_surface(
                points=np.array(ordered_positions, dtype=np.float32)
            ).tolist()
        )

        corner_updates = zip(
            ordered_corners, ordered_position_in_surface_space_undistorted
        )

        for (corner, new_undistorted) in corner_updates:
            # TODO: Provide Surface API for moving multiple corners in one call
            surface._move_corner(
                corner=corner, new_position_in_surface_space_undistorted=new_undistorted
            )

    def add_markers_to_surface(
        self,
        surface: Surface,
        location: SurfaceLocation,
        markers: T.List[Marker],
        ignore_location_staleness: bool = False,
    ):
        """ """

        # Validate the surface definition and the surface location argumentse
        self.__argument_validator.validate_surface_and_location(
            surface=surface,
            location=location,
            ignore_location_staleness=ignore_location_staleness,
        )

        # Ensure marker uniqueness
        marker_uids = {m.uid for m in markers}

        # Ensure only markers that are not part of the definition will be added
        marker_uids = marker_uids.difference(surface.registered_marker_uids)

        # Filter out unused markers
        markers = [m for m in markers if m.uid in marker_uids]

        if len(markers) == 0:
            # If there are no markers to add, return without invalidating the locations
            return

        # Since this action mutates the surface definition, mark previously computed
        # locations as stale
        self.__locations_tracker.mark_locations_as_stale_for_surface(surface=surface)

        for marker in markers:

            marker_undistorted = location._map_marker_from_image_to_surface(
                marker=marker
            )

            surface._add_marker(marker_undistorted=marker_undistorted)

    def remove_markers_from_surface(
        self,
        surface: Surface,
        location: SurfaceLocation,
        marker_uids: T.List[MarkerId],
        ignore_location_staleness: bool = False,
    ):
        """ """

        # Validate the surface definition and the surface location arguments
        self.__argument_validator.validate_surface_and_location(
            surface=surface,
            location=location,
            ignore_location_staleness=ignore_location_staleness,
        )

        # Ensure marker uniqueness
        marker_uids = set(marker_uids)

        # Ensure only markers that are part of the definition will be removed
        marker_uids = marker_uids.intersection(surface.registered_marker_uids)

        if len(marker_uids) == 0:
            # If there are no markers to remove, return without invalidating the
            # locations
            return

        # Since this action mutates the surface definition, mark previously computed
        # locations as stale
        self.__locations_tracker.mark_locations_as_stale_for_surface(surface=surface)

        for marker_uid in marker_uids:
            surface._remove_marker(marker_uid=marker_uid)

    def set_orientation(
        self, surface: Surface, starting_with: CornerId, clockwise: bool = True
    ):
        # Validate the surface definition
        self.__argument_validator.validate_surface(surface=surface)

        orientation = SurfaceOrientation(starting_with, clockwise)
        surface.orientation = orientation

    # ## Locating a surface

    def get_relative_rotation(self, surface: Surface) -> int:
        # Validate the surface definition
        self.__argument_validator.validate_surface(surface=surface)

        return surface.orientation.get_relative_rotation()

    def locate_surface(
        self, surface: Surface, markers: T.List[Marker]
    ) -> T.Optional[SurfaceLocation]:
        """Computes a SurfaceLocation based on a list of visible markers"""

        # Validate the surface definition
        self.__argument_validator.validate_surface(surface=surface)

        location = SurfaceLocation._create_location_from_markers(
            surface=surface, markers=markers
        )

        if location is not None:
            # Track the created location to invalidate it if the surface is mutated
            # later
            self.__locations_tracker.track_location_for_surface(
                surface=surface, location=location
            )

        return location

    def locate_surface_visual_anchors(
        self, surface: Surface, location: SurfaceLocation
    ) -> T.Optional[SurfaceVisualAnchors]:
        """ """

        # Validate the surface definition and the surface location arguments
        self.__argument_validator.validate_surface_and_location(
            surface=surface, location=location, ignore_location_staleness=False
        )

        return SurfaceVisualAnchors._create_from_location(
            location=location, orientation=surface.orientation
        )

    def locate_surface_image_crop(
        self,
        surface: Surface,
        location: SurfaceLocation,
        camera: Camera,
        width: T.Optional[int] = None,
        height: T.Optional[int] = None,
    ) -> SurfaceImageCrop:
        """ """

        # Validate the surface definition and the surface location arguments
        self.__argument_validator.validate_surface_and_location(
            surface=surface, location=location, ignore_location_staleness=False
        )

        # swap width and height if surface location is turned by 90 degree in comparison
        # to internal surface representation
        if surface.orientation.get_relative_rotation() % 180 != 0:
            temp = width
            width = height
            height = temp

        return SurfaceImageCrop._create_image_crop(
            location, camera, width=width, height=height
        )

    def locate_surface_image_crop_with_heatmap(
        self,
        surface: Surface,
        location: SurfaceLocation,
        camera: Camera,
        points: T.List[T.Tuple[int, int]],
        width: T.Optional[int] = None,
        height: T.Optional[int] = None,
    ) -> (SurfaceImageCrop, SurfaceHeatmap):
        """ """

        # Validate the surface definition and the surface location arguments
        self.__argument_validator.validate_surface_and_location(
            surface=surface, location=location, ignore_location_staleness=False
        )

        image_crop = self.locate_surface_image_crop(
            surface=surface,
            location=location,
            camera=camera,
            width=width,
            height=height,
        )

        heatmap = SurfaceHeatmap._create_surface_heatmap(
            points_in_image_space=points, location=location
        )

        return (image_crop, heatmap)


# #### Private Helpers


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
        """Validate the standalone `surface` argument"""

        if not isinstance(surface, Surface):
            raise ValueError(
                f'Expected an instance of Surface, but got "{surface.__class__}"'
            )

    @staticmethod
    def validate_surface_and_location(
        surface: Surface, location: SurfaceLocation, ignore_location_staleness: bool
    ):
        """Validate the pair of `surface` and `location` arguments"""

        _SurfaceTrackerArgumentValidator.validate_surface(surface=surface)

        if not isinstance(location, SurfaceLocation):
            raise ValueError(
                f'Expected an instance of SurfaceLocation, but got "{location.__class__}"'
            )

        if surface.uid != location.surface_uid:
            raise ValueError(f"SurfaceId missmatch: location doesn't belong to surface")

        if (not ignore_location_staleness) and location.is_stale:
            raise ValueError(
                f"Stale location: the surface definition has changed; location must be recomputed"
            )
