import typing as T

import numpy as np

from .surface import Surface


class GazeOnSurfaceDistribution:
    def __init__(
        self,
        gaze: list,
        surface: Surface,
        heatmap_size: T.Tuple[int, int],
    ):
        # TODO: Type annotate `gaze`
        raise NotImplementedError()

    def heatmap(self) -> np.ndarray:
        raise NotImplementedError()
