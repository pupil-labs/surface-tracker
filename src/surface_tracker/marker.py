import abc
import typing as T

from .corner import CornerId


MarkerId = T.NewType("MarkerId", str)


VertexInImageSpace = T.Tuple[int, int]


class Marker(abc.ABC):

    ### Abstract members

    @property
    @abc.abstractmethod
    def uid(self) -> MarkerId:
        raise NotImplementedError()

    @abc.abstractmethod
    def vertices(self) -> T.List[tuple]:
        # TODO: Add option to explicitly define the order of the vertices list
        # e.g.: marker.vertices_in_image_space(starting_with=CornerId.BOTTOM_RIGHT, clockwise=False)
        raise NotImplementedError()

    ### Serialize

    @staticmethod
        raise NotImplementedError()
    @abc.abstractmethod
    def from_dict(value: dict) -> "Marker":

    @abc.abstractmethod
    def as_dict(self) -> dict:
        raise NotImplementedError()
