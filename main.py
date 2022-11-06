import numpy as np

from calib_utils import *


if __name__ == '__main__':
    # Get model points X and associated sensor points U in M views
    root_path = './chessboard_data'
    model_points, sensor_points = find_chessboard_corners(root_path, use_sub_pix=True, show=False)

    # np.savetxt('./test/model.txt', model_points[0], delimiter=' ')
    # for i in range(len(sensor_points)):
    #     np.savetxt('./test/data' + str(i).zfill(2) + '.txt', sensor_points[i], delimiter=' ')

    # calibrate the camera
    A, k, W = calibrate(model_points, sensor_points)
    # calibrate(model_points, sensor_points)

    # print('Intrinsics:\n', A)
    # print('Distortion:\n', k)
