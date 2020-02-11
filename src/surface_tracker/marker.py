import typing as T


MarkerId = T.NewType("MarkerId", str)


class Marker:
    def __init__(self, uid: MarkerId, vert):
        # TODO: Type annotate `vert`
        # TODO: Come up with a more descriptive name for `vert`
        self.__uid = uid
        self.__vert = vert

    @property
    def uid(self) -> MarkerId:
        return self.__uid

    @property
    def vert(self):
        # TODO: Type annotate `vert`
        # TODO: Come up with a more descriptive name for `vert`
        return self.__vert

    ### Serialize

    @staticmethod
    def from_dict(self) -> dict:
        raise NotImplementedError()

    def as_dict(self) -> dict:
        raise NotImplementedError()
