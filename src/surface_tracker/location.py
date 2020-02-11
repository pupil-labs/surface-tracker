import typing as T

import numpy as np

from .camera import CameraModel
from .surface import Surface, SurfaceId


class SurfaceLocation:
    def __init__(
        self,
        surface_uid: SurfaceId,
        detected_markers_count: int,
        dist_img_to_surf_trans,
        surf_to_dist_img_trans,
        img_to_surf_trans,
        surf_to_img_trans,
    ):
        raise NotImplementedError()

    ### Mapping

    def map_from_image_to_surface(
        self,
        points: np.ndarray,
        camera_model: CameraModel,
        compensate_distortion: bool = True,
        transform_matrix=None,
    ) -> np.ndarray:
        raise NotImplementedError()

    def map_from_surface_to_image(
        self,
        points: np.ndarray,
        camera_model: CameraModel,
        compensate_distortion: bool = True,
        transform_matrix=None,
    ) -> np.ndarray:
        raise NotImplementedError()

    ### Serialize

    def as_dict(self) -> dict:
        raise NotImplementedError()

    @staticmethod
    def from_dict(value: dict) -> "SurfaceLocation":
        raise NotImplementedError()
