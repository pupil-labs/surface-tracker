import enum
import typing as T

from . import utils


class CornerId(enum.Enum):
    TOP_LEFT = (0, 0)
    TOP_RIGHT = (1, 0)
    BOTTOM_RIGHT = (1, 1)
    BOTTOM_LEFT = (0, 1)

    @staticmethod
    def all_corners(starting_with: "CornerId" = None, clockwise: bool = True):
        """
        Enumerate all corners.

        Args:
            starting_with: First `CornerId` in the returned list.
            clockwise: Direction of the returned list.

        Returns:
            List of `CornerId`
        """
        starting_with = CornerId.TOP_LEFT if starting_with is None else starting_with
        corners = list(CornerId)

        if not clockwise:
            corners.reverse()

        n = corners.index(starting_with)
        corners = utils.left_rotation(corners, n)

        return corners

    ### Serialize

    @staticmethod
    def from_tuple(value: T.Tuple[float, float]):
        return CornerId(value)

    def as_tuple(self) -> T.Tuple[float, float]:
        return tuple(self.value)