import typing as T

import numpy as np

from .surface import Surface


class GazeOnSurface:
    def __init__(
        self,
        gaze: list,
        surface: Surface,
        world_size: T.Tuple[int, int],
        heatmap_size: T.Tuple[int, int],
    ):
        # TODO: Type annotate `gaze`
        self.__world_size = world_size
        raise NotImplementedError()

    @property
    def world_size(self) -> T.Tuple[int, int]:
        return self.__world_size

    def heatmap(self) -> np.ndarray:
        raise NotImplementedError()
