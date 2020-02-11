import typing as T

from .camera import CameraModel
from .marker import Marker, MarkerId
from .location import SurfaceLocation
from .surface import Surface, SurfaceId


class SurfaceTracker:
    def create_surface(
        self, name: str, markers: T.List[Marker], camera_model: CameraModel
    ) -> Surface:
        raise NotImplementedError()

    def locate_surface(
        self, surface: Surface, markers: T.List[Marker], camera_model: CameraModel
    ) -> T.Optional[SurfaceLocation]:
        raise NotImplementedError()
