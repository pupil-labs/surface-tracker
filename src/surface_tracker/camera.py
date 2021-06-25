"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs
Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See LICENSE for license details.
---------------------------------------------------------------------------~(*)
"""

import cv2
import numpy as np


class Camera:
    def __init__(
        self, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray
    ) -> np.ndarray:
        self.camera_matrix = np.asarray(camera_matrix)
        self.distortion_coefficients = np.asarray(distortion_coefficients)

    def distort_and_project(self, points: np.ndarray):
        points.shape = -1, 1, 2

        points_homogeneous = cv2.convertPointsToHomogeneous(points)
        proj_and_dist_points = np.squeeze(
            cv2.projectPoints(
                points_homogeneous,
                rvec=np.zeros((1, 3)),
                tvec=np.zeros((1, 3)),
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.distortion_coefficients,
            )[0]
        )
        return proj_and_dist_points
