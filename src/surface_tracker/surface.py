import abc
import enum
import typing as T

from .corner import CornerId
from .marker import Marker, MarkerId


SurfaceId = T.NewType("SurfaceId", str)


class Surface(abc.ABC):

    ### Info

    @property
    @abc.abstractmethod
    def uid(self) -> SurfaceId:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def version(self) -> int:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def world_size(self) -> T.Tuple[int, int]:
        raise NotImplementedError()

    ### Update

    @abc.abstractmethod
    def add_marker(self, marker: Marker):
        raise NotImplementedError()

    @abc.abstractmethod
    def remove_marker(self, marker_uid: MarkerId):
        raise NotImplementedError()

    @abc.abstractmethod
    def move_corner(self, corner_uid: CornerId, new_position):
        # TODO: Type annotate new_position
        raise NotImplementedError()

    ### Serialize

    @staticmethod
    @abc.abstractmethod
    def from_dict(value: dict) -> "Surface":
        raise NotImplementedError()

    @abc.abstractmethod
    def as_dict(self) -> dict:
        raise NotImplementedError()
