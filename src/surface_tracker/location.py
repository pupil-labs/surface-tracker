import typing as T

import numpy as np

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

    ### Serialize

    def as_dict(self) -> dict:
        raise NotImplementedError()

    @staticmethod
    def from_dict(value: dict) -> "SurfaceLocation":
        raise NotImplementedError()
