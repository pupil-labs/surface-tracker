import typing as T

import numpy as np

from .surface import Surface


class GazeOnSurfaceDistribution:
    def __init__(
        self,
        surface: Surface,
        heatmap_size: T.Tuple[int, int],
        gaze: T.Optional[list] = None,
    ):
        raise NotImplementedError()

    def update(self, gaze: T.Optional[list]):
        # TODO: Type annotate `gaze`
        raise NotImplementedError()

    def heatmap(self) -> np.ndarray:
        raise NotImplementedError()
