import typing as T

import numpy as np
import cv2


def left_rotation(a: list, k: int):
    """Rotate list to the left
    e.g.: [1, 2, 3, 4] -> [2, 3, 4, 1]
    """
    # if the size of k > len(a), rotate only necessary with
    # # module of the division
    rotations = k % len(a)
    return a[rotations:] + a[:rotations]


def right_rotation(a: list, k: int):
    """Rotate list to the right
    e.g.: [1, 2, 3, 4] -> [4, 1, 2, 3]
    """
    # if the size of k > len(a), rotate only necessary with
    # module of the division
    rotations = k % len(a)
    return a[-rotations:] + a[:-rotations]


def bounding_quadrangle(vertices: np.ndarray):

    # According to OpenCV implementation, cv2.convexHull only accepts arrays with
    # 32bit floats (CV_32F) or 32bit signed ints (CV_32S).
    # See: https://github.com/opencv/opencv/blob/3.4/modules/imgproc/src/convhull.cpp#L137
    # See: https://github.com/pupil-labs/pupil/issues/1544
    vertices = np.asarray(vertices, dtype=np.float32)

    hull_points = cv2.convexHull(vertices, clockwise=False)

    # The convex hull of a list of markers must have at least 4 corners, since a
    # single marker already has 4 corners. If the convex hull has more than 4
    # corners we reduce that number with approximations of the hull.
    if len(hull_points) > 4:
        new_hull = cv2.approxPolyDP(hull_points, epsilon=1, closed=True)
        if new_hull.shape[0] >= 4:
            hull_points = new_hull

    if len(hull_points) > 4:
        curvature = abs(GetAnglesPolyline(hull_points, closed=True))
        most_acute_4_threshold = sorted(curvature)[3]
        hull_points = hull_points[curvature <= most_acute_4_threshold]

    # Vertices space is flipped in y.  We need to change the order of the
    # hull_points vertices
    hull_points = hull_points[[1, 0, 3, 2], :, :]

    # Roll the hull_points vertices until we have the right orientation:
    # vertices space has its origin at the image center. Adding 1 to the
    # coordinates puts the origin at the top left.
    distance_to_top_left = np.sqrt(
        (hull_points[:, :, 0] + 1) ** 2 + (hull_points[:, :, 1] + 1) ** 2
    )
    bot_left_idx = np.argmin(distance_to_top_left) + 1
    hull_points = np.roll(hull_points, -bot_left_idx, axis=0)
    return hull_points


# From pupil_src/shared_modules/methods.py
def GetAnglesPolyline(polyline, closed=False):
    """
    see: http://stackoverflow.com/questions/3486172/angle-between-3-points
    ported to numpy
    returns n-2 signed angles
    """

    points = polyline[:, 0]

    if closed:
        a = np.roll(points, 1, axis=0)
        b = points
        c = np.roll(points, -1, axis=0)
    else:
        a = points[0:-2]  # all "a" points
        b = points[1:-1]  # b
        c = points[2:]  # c points
    # ab =  b.x - a.x, b.y - a.y
    ab = b - a
    # cb =  b.x - c.x, b.y - c.y
    cb = b - c
    # float dot = (ab.x * cb.x + ab.y * cb.y); # dot product
    # print 'ab:',ab
    # print 'cb:',cb

    # float dot = (ab.x * cb.x + ab.y * cb.y) dot product
    # dot  = np.dot(ab,cb.T) # this is a full matrix mulitplication we only need the diagonal \
    # dot = dot.diagonal() #  because all we look for are the dotproducts of corresponding vectors (ab[n] and cb[n])
    dot = np.sum(
        ab * cb, axis=1
    )  # or just do the dot product of the correspoing vectors in the first place!

    # float cross = (ab.x * cb.y - ab.y * cb.x) cross product
    cros = np.cross(ab, cb)

    # float alpha = atan2(cross, dot);
    alpha = np.arctan2(cros, dot)
    return alpha * (180.0 / np.pi)  # degrees
    # return alpha #radians
