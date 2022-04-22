"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs
Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See LICENSE for license details.
---------------------------------------------------------------------------~(*)
"""
import typing as T

import cv2
import numpy as np

from .camera import Camera
from .corner import CornerId
from .location import SurfaceLocation


# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
class SurfaceImageCrop:
    @staticmethod
    def _create_image_crop(
        location: SurfaceLocation,
        camera: Camera,
        width: T.Optional[int],
        height: T.Optional[int],
    ):
        surface_corners_in_surface_space = CornerId.all_corners(
            starting_with=CornerId.TOP_LEFT, clockwise=True
        )
        surface_corners_in_surface_space = [
            c.as_tuple() for c in surface_corners_in_surface_space
        ]
        surface_corners_in_surface_space = np.array(
            surface_corners_in_surface_space, dtype=np.float32
        )

        surface_corners_in_image_space = location._map_from_surface_to_image(
            points=surface_corners_in_surface_space
        )

        surface_corners_in_image_space_distorted = camera.distort_and_project(
            points=surface_corners_in_image_space
        )

        crop_size = SurfaceImageCrop.__calculate_crop_size(
            *surface_corners_in_image_space_distorted, width=width, height=height
        )
        crop_w, crop_h = crop_size

        crop_corners_in_image_space = [
            [0, 0],
            [crop_w - 1, 0],
            [crop_w - 1, crop_h - 1],
            [0, crop_h - 1],
        ]
        crop_corners_in_image_space = np.array(
            crop_corners_in_image_space, dtype=np.float32
        )
        surface_corners_in_image_space_distorted = np.array(
            surface_corners_in_image_space_distorted, dtype=np.float32
        )

        perspective_transform = cv2.getPerspectiveTransform(
            surface_corners_in_image_space_distorted, crop_corners_in_image_space
        )

        return SurfaceImageCrop(
            crop_size_in_image_space=crop_size,
            perspective_transform=perspective_transform,
        )

    def __init__(
        self,
        crop_size_in_image_space: T.Tuple[int, int],
        perspective_transform: np.ndarray,
    ):
        self.__crop_size_in_image_space = crop_size_in_image_space
        self.__perspective_transform = perspective_transform

    @property
    def size_in_image_space(self) -> T.Tuple[int, int]:
        return self.__crop_size_in_image_space

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        surface_image = cv2.warpPerspective(
            image, self.__perspective_transform, self.__crop_size_in_image_space
        )
        surface_image = self.__flip_image_vertically(
            surface_image
        )  # Flip image vertically #FIXME: ???
        return surface_image

    @staticmethod
    def __flip_image_vertically(image):
        h = image.shape[0]
        flipped = np.zeros(image.shape, np.uint8)
        for i in range(h):
            flipped[i, :] = image[h - i - 1, :]
        return flipped

    @staticmethod
    def __calculate_crop_size(
        tl: int,
        tr: int,
        br: int,
        bl: int,
        width: T.Optional[int],
        height: T.Optional[int],
    ) -> T.Tuple[int, int]:

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(width_a, width_b)

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(height_a, height_b)

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order

        if width is not None and height is not None:
            raise ValueError(f'Expected only "width" OR "height" to be supplied')

        if width is not None:
            ratio = max_width / width
            max_width = width
            max_height /= ratio

        if height is not None:
            ratio = max_height / height
            max_height = height
            max_width /= ratio

        return int(max_width), int(max_height)
