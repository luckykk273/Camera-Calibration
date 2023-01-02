from utils.calibration import find_chessboard_corners, calibrate


if __name__ == '__main__':
    # Get model points X and associated sensor points U in M views
    root_path = './chessboard_data'
    model_points, sensor_points = find_chessboard_corners(root_path, use_sub_pix=True, show=False)

    # calibrate the camera
    # NOTE: Model points in all views should be the same, so we just need to pass in one.
    A, k, W = calibrate(model_points[0], sensor_points)

    print('Focal Length   : (%.6f, %.6f)' % (A[0, 0], A[1, 1]))
    print('Principal Point: (%.6f, %.6f)' % (A[0, 2], A[1, 2]))
    print('Skew           : %.6f' % A[0, 1])
    print('Distortion     : (%.6f, %.6f)' % (k[0], k[1]))
