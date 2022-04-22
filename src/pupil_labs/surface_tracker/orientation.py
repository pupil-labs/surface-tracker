"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs
Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See LICENSE for license details.
---------------------------------------------------------------------------~(*)
"""
from .corner import CornerId


class SurfaceOrientation:
    def __init__(
        self, surface_start_with: CornerId = CornerId.TOP_LEFT, clockwise: bool = True
    ):
        self.__surface_start_with = surface_start_with
        self.__clockwise = clockwise
        if not clockwise:
            raise NotImplementedError()

    @property
    def surface_start_with(self):
        return self.__surface_start_with

    @property
    def clockwise(self):
        return self.__clockwise

    def map_surface_corners_to_rotation(self, corner_dict):
        mapping = self.__mapping_default_to_rotated()
        return self.__map_surface_corners(corner_dict, mapping)

    def map_surface_corners_to_default(self, corner_dict):
        mapping = self.__mapping_rotated_to_default()
        return self.__map_surface_corners(corner_dict, mapping)

    def __mapping_default_to_rotated(self):
        default_orientation = CornerId.all_corners()
        rotated_orientation = CornerId.all_corners(
            starting_with=self.surface_start_with, clockwise=self.clockwise
        )
        return dict(zip(default_orientation, rotated_orientation))

    def __mapping_rotated_to_default(self):
        default_orientation = CornerId.all_corners()
        rotated_orientation = CornerId.all_corners(
            starting_with=self.surface_start_with, clockwise=self.clockwise
        )
        return dict(zip(rotated_orientation, default_orientation))

    def __map_surface_corners(self, corner_dict, mapping):
        updated_corner_dict = {}
        for corner_name, coords in corner_dict.items():
            updated_corner_dict[mapping[corner_name]] = coords
        return updated_corner_dict

    def get_relative_rotation(self, clockwise=True):
        default_start_with = CornerId.all_corners()[0]
        rotated_orientation = CornerId.all_corners(
            starting_with=self.surface_start_with, clockwise=self.clockwise
        )
        clockwise_rotation = rotated_orientation.index(default_start_with) * 90
        if not clockwise:
            return -clockwise_rotation % 360
        return clockwise_rotation

    def get_visual_anchor_surface_space(self):
        if self.surface_start_with == CornerId.TOP_LEFT and self.clockwise:
            top_indicator_corners = [[0.3, 0.7], [0.7, 0.7], [0.5, 0.9]]
            top_indicator_corners.append(top_indicator_corners[0])
        if self.surface_start_with == CornerId.TOP_RIGHT and self.clockwise:
            top_indicator_corners = [[0.7, 0.7], [0.7, 0.3], [0.9, 0.5]]
            top_indicator_corners.append(top_indicator_corners[0])
        if self.surface_start_with == CornerId.BOTTOM_RIGHT and self.clockwise:
            top_indicator_corners = [[0.7, 0.3], [0.3, 0.3], [0.5, 0.1]]
            top_indicator_corners.append(top_indicator_corners[0])
        if self.surface_start_with == CornerId.BOTTOM_LEFT and self.clockwise:
            top_indicator_corners = [[0.3, 0.3], [0.3, 0.7], [0.1, 0.5]]
            top_indicator_corners.append(top_indicator_corners[0])
        return top_indicator_corners

    def as_dict(self) -> dict:
        return {
            "surface_start_with": self.surface_start_with.name,
            "clockwise": self.clockwise,
        }

    @staticmethod
    def from_dict(value: dict) -> "SurfaceOrientation":
        try:
            return SurfaceOrientation(
                surface_start_with=CornerId.from_name(value["surface_start_with"]),
                clockwise=value["clockwise"],
            )
        except Exception as err:
            raise ValueError(err)
