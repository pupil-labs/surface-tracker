import enum


class CoordinateSpace(enum.Enum):
    IMAGE_DISTORTED = "image-distorted"
    IMAGE_UNDISTORTED = "image-undistorted"
    SURFACE_DISTORTED = "surface-distorted"
    SURFACE_UNDISTORTED = "surface-undisitorted"
