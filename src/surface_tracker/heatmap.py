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
import typing as T

import cv2
import numpy as np

from .location import SurfaceLocation
from .surface import SurfaceId


class SurfaceHeatmap:
    class ColorFormat(enum.Enum):
        RGB = enum.auto()

        @property
        def channel_count(self) -> int:
            if self == SurfaceHeatmap.ColorFormat.RGB:
                return 3
            raise NotImplementedError()

    @staticmethod
    def _create_surface_heatmap(
        points_in_image_space: T.List[T.Tuple[int, int]], location: SurfaceLocation
    ):
        surface_heatmap = SurfaceHeatmap(surface_uid=location.surface_uid)
        surface_heatmap._add_points(
            points_in_image_space=points_in_image_space, location=location
        )
        return surface_heatmap

    def __init__(self, surface_uid: SurfaceId):
        self.__surface_uid = surface_uid
        self.__points_in_surface_space_numpy = []
        self._invalidate_cached_computations()

    def _add_points(
        self,
        points_in_image_space: T.List[T.Tuple[int, int]],
        location: SurfaceLocation,
    ):
        points_in_image_space_numpy = np.asarray(
            points_in_image_space, dtype=np.float32
        )
        new_points_in_surface_space_numpy = location._map_from_image_to_surface(
            points=np.asarray(points_in_image_space_numpy, dtype=np.float32)
        )

        for p in new_points_in_surface_space_numpy:
            self.__points_in_surface_space_numpy.append(p)

    def image(
        self,
        size: T.Tuple[int, int],
        color_format: T.Optional["SurfaceHeatmap.ColorFormat"] = None,
    ) -> np.ndarray:

        if color_format is None:
            color_format = SurfaceHeatmap.ColorFormat.RGB
        elif not isinstance(color_format, SurfaceHeatmap.ColorFormat):
            raise ValueError(
                f"color_format must be an instance of SurfaceHeatmap.ColorFormat"
            )

        cache_key = (size, color_format)
        heatmap_resolution = 31
        heatmap_blur_factor = 0.0

        if cache_key not in self.__heatmap_image_by_size_and_color_format:
            heatmap_data = np.asarray(self.__points_in_surface_space_numpy)

            aspect_ratio = size[1] / size[0]
            grid = (
                max(1, int(heatmap_resolution * aspect_ratio)),
                int(heatmap_resolution),
            )

            if len(heatmap_data) > 0:
                xvals = heatmap_data[:, 0]
                yvals = 1.0 - heatmap_data[:, 1]
                histogram, *edges = np.histogram2d(
                    yvals, xvals, bins=grid, range=[[0, 1.0], [0, 1.0]], normed=False
                )
                filter_h = 19 + heatmap_blur_factor * 15
                filter_w = filter_h * aspect_ratio
                filter_h = int(filter_h) // 2 * 2 + 1
                filter_w = int(filter_w) // 2 * 2 + 1

                histogram = cv2.GaussianBlur(histogram, (filter_h, filter_w), 0)
                histogram_max = histogram.max()
                histogram *= (255.0 / histogram_max) if histogram_max else 0.0
                histogram = histogram.astype(np.uint8)
            else:
                histogram = np.zeros(grid, dtype=np.uint8)

            histogram = cv2.applyColorMap(histogram, cv2.COLORMAP_JET)

            if color_format == SurfaceHeatmap.ColorFormat.RGB:
                heatmap = np.ones((*grid, color_format.channel_count), dtype=np.uint8)
                heatmap[:, :, 0] = histogram[:, :, 2]  # red
                heatmap[:, :, 1] = histogram[:, :, 1]  # green
                heatmap[:, :, 2] = histogram[:, :, 0]  # blue
            else:
                raise ValueError(f'Unsupported color_format: "{color_format}"')

            heatmap = cv2.resize(heatmap, dsize=size, interpolation=cv2.INTER_CUBIC)

            assert len(heatmap.shape) == 3  # sanity check
            assert (heatmap.shape[1], heatmap.shape[0]) == size  # sanity check
            self.__heatmap_image_by_size_and_color_format[cache_key] = heatmap

        return self.__heatmap_image_by_size_and_color_format[cache_key]

    def _invalidate_cached_computations(self):
        self.__heatmap_image_by_size_and_color_format = {}
