import cv2 as cv
import glob
import numpy as np
import os

from .homography_utils import get_homographies
from .intrinsics_utils import get_camera_intrinsics
from .extrinsics_utils import get_extrinsics
from .distortion_utils import estimate_lens_distortion


def find_chessboard_corners(root_path='./chessboard_data',
                            pattern_size=(7, 6),
                            square_size=1.0,
                            use_sub_pix=False,
                            show=False):
    """
    Find chessboard corners(model points) X and associated sensor points U.

    NOTE
    ----
    In this repository, to follow the terms in reference[2],
    we use `model points` for points in 3D world coordinate and `sensor points` for points in 2D sensor coordinate.

    But in OpenCV tutorial, it uses `object points` and `image points`.

    They represent the same things.


    :param root_path: the root path where the model data in.
    :param pattern_size: points per row and points per column on chessboard which equal to (columns, rows)
    :param square_size: The physical size of the chessboard square
    :param use_sub_pix: To use sub pixel to refine the location of sensor points or not
    :param show: show the found corners in each image or not

    :return: model points and sensor points
        model points: (X, Y) in world coordinate
        sensor points: (u, v) in sensor coordinate
    """
    row, col = pattern_size[1], pattern_size[0]

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare model points, like (0, 0), (1, 0), (2, 0) ....,(6, 5)
    model_pts = np.mgrid[0:col, 0:row].T.reshape(-1, 2).astype(np.float32)
    model_pts *= square_size

    # Arrays to store model points and sensor points from all the images.
    model_points = []  # 3d points in world coordinate
    sensor_points = []  # 2d points in sensor coordinate

    images = glob.glob(root_path + os.sep + '*.jpg')
    for img_name in images:
        img = cv.imread(img_name)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

        # If found, add model points, sensor points
        if ret:
            model_points.append(model_pts)
            if use_sub_pix:
                corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            sensor_points.append(np.squeeze(corners, axis=1))
            if show:
                # Draw and display the corners
                cv.drawChessboardCorners(img, pattern_size, corners, ret)
                cv.imshow('img', img)
                cv.waitKey(500)
                cv.destroyAllWindows()
        else:
            print('Error in detection points:', img_name)

    return model_points, sensor_points


def calibrate(Xs: list, Us: list):
    """
    Camera calibration algorithm by Zhang.
    All variable names follow pseudocode summarized in Section 4 in reference [2].

    :param Xs: an ordered sequence of 3D points on the planar target with Xj = (Xj, Yj)^T
    :param Us: a sequence of views, each view is an ordered sequence of observed image points uij = (uij, vij)^T
    :return: the estimated intrinsics parameters A, k and the extrinsic transformations W = (R | t) for each view.
    """
    H_init = get_homographies(Xs, Us)
    A_init = get_camera_intrinsics(H_init)
    W_init = get_extrinsics(A_init, H_init)
    k_init = estimate_lens_distortion(A_init, W_init, Xs, Us)
    # A, k, W = refine_all(A_init, k_init, W_init, Xs, Us)
    # return A, k, W
    return H_init, A_init, W_init
