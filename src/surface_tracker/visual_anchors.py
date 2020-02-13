import typing as T

import numpy as np
import cv2

from .camera import CameraModel
from .corner import CornerId
from .location import SurfaceLocation
from .surface import Surface, SurfaceId


class SurfaceVisualAnchors:
    @staticmethod
    def _create_from_location(
        location: SurfaceLocation, camera_model: CameraModel
    ) -> "SurfaceVisualAnchors":

        perimeter_corners = CornerId.all_corners()
        perimeter_corners = [c.as_tuple() for c in perimeter_corners]
        perimeter_corners.append(perimeter_corners[0])

        top_indicator_corners = [[0.3, 0.7], [0.7, 0.7], [0.5, 0.9]]
        top_indicator_corners.append(top_indicator_corners[0])

        perimeter_points_surface_space = np.array(perimeter_corners, dtype=np.float32)
        perimeter_points_image_space = location.map_from_surface_to_image(
            points=perimeter_points_surface_space,
            camera_model=camera_model,
            compensate_distortion=False,
        )

        top_indicator_points_in_surface_space = np.array(
            top_indicator_corners, dtype=np.float32
        )
        top_indicator_points_in_image_space = location.map_from_surface_to_image(
            points=top_indicator_points_in_surface_space,
            camera_model=camera_model,
            compensate_distortion=False,
        )

        title_anchor = perimeter_points_image_space.reshape((5, -1))[2]
        title_anchor = title_anchor[0], title_anchor[1] - 75

        edit_surface_anchor = title_anchor[0], title_anchor[1] + 25
        edit_markers_anchor = title_anchor[0], title_anchor[1] + 50

        return SurfaceVisualAnchors(
            top_polyline=top_indicator_points_in_image_space.reshape((4, 2)).tolist(),
            perimeter_polyline=perimeter_points_image_space.reshape((5, 2)).tolist(),
            title_anchor=title_anchor,
            edit_surface_anchor=edit_surface_anchor,
            edit_markers_anchor=edit_markers_anchor,
        )

    def __init__(
        self,
        top_polyline: T.List[T.Tuple[float, float]],
        perimeter_polyline: T.List[T.Tuple[float, float]],
        title_anchor: T.Tuple[float, float],
        edit_surface_anchor: T.Tuple[float, float],
        edit_markers_anchor: T.Tuple[float, float],
    ):
        self.__top_polyline = top_polyline
        self.__perimeter_polyline = perimeter_polyline
        self.__title_anchor = title_anchor
        self.__edit_surface_anchor = edit_surface_anchor
        self.__edit_markers_anchor = edit_markers_anchor

    @property
    def top_polyline(self) -> T.List[T.Tuple[float, float]]:
        return list(self.__top_polyline)

    @property
    def perimeter_polyline(self) -> T.List[T.Tuple[float, float]]:
        return list(self.__perimeter_polyline)

    @property
    def title_anchor(self) -> T.Tuple[float, float]:
        return tuple(self.__title_anchor)

    @property
    def edit_surface_anchor(self) -> T.Tuple[float, float]:
        return tuple(self.__edit_surface_anchor)

    @property
    def edit_markers_anchor(self) -> T.Tuple[float, float]:
        return tuple(self.__edit_markers_anchor)
