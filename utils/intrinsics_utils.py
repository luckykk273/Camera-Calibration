from math_utils import solve, v_pq
import numpy as np


def get_camera_intrinsics(H):
    """
    Extract intrinsics parameters from homography matrices.

    Here we implement three methods to compare the result:
        1. the version A defined in algorithm 4.4 in reference [2]
        2. the version A defined in algorithm 4.4 in reference [2] but replace the equations to the definition in reference [1]
        3. the version B defined in algorithm 4.5 in reference [2]

    By default, we return the result computed by method 3.

    :param H: a sequence of homography matrices
    :return: the common intrinsics matrix
    """
    M = len(H)
    V = np.zeros((2 * M, 6))
    for i in range(M):
        V[2 * i, :] = v_pq(H[i], 0, 1)
        V[2 * i + 1, :] = v_pq(H[i], 0, 0) - v_pq(H[i], 1, 1)

    # In reference [1], it is mentioned that if M = 2, we should impose the skewness constraint: gamma = 0 because
    # we can't use only 2M = 4 observations to solve a linear system with dof = 5.
    if M == 2:
        V[-1, :] = [0, 1, 0, 0, 0, 0]

    b = solve(V)

    # method 1: version A defined in algorithm 4.4 in reference [2]
    # TODO: the gamma parameter is different from method 2 and 3(method 2 and 3 are the same);
    #       check if something wrong?
    w = b[0] * b[2] * b[5] - b[1]**2 * b[5] - b[0] * b[4]**2 + 2 * b[1] * b[3] * b[4] - b[2] * b[3]**2
    d = b[0] * b[2] - b[1]**2
    alpha = np.sqrt(w / (d * b[0]))
    beta = np.sqrt(w / d**2 * b[0])
    gamma = np.sqrt(w / (d**2 * b[0])) * b[1]
    uc = (b[1] * b[4] - b[2] * b[3]) / d
    vc = (b[1] * b[3] - b[0] * b[4]) / d
    A1 = np.array([
        [alpha, gamma, uc],
        [    0,  beta, vc],
        [    0,     0,  1]
    ])

    # used by method 2 and 3
    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])

    # method 2: version A defined in algorithm 4.4 in reference [2]
    #           but replace the equations to the definition in reference [1]
    v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1]**2)
    lambda_ = B[2, 2] - (B[0, 2]**2 + v0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
    alpha = np.sqrt(lambda_ / B[0, 0])
    beta = np.sqrt(lambda_ * B[0, 0] / (B[0, 0] * B[1, 1] - B[0, 1]**2))
    gamma = -B[0, 1] * alpha**2 * beta / lambda_
    u0 = gamma * v0 / beta - B[0, 2] * alpha**2 / lambda_
    A2 = np.array([
        [alpha, gamma, u0],
        [    0,  beta, v0],
        [    0,     0,  1]
    ])

    # method 3: version B(Cholesky decomposition) defined in algorithm 4.5 in reference [2]
    try:
        L = np.linalg.cholesky(B)
    except np.linalg.LinAlgError as e:
        print('B is not positive definite. \n'
              'Try -B.')
        L = np.linalg.cholesky(-B)
    A3 = L[2, 2] * np.linalg.inv(L).T

    # if A1 - A2 > 1e-6:
    #     raise ValueError('Numerical error occurred.'
    #                      'Version A: %s\n'
    #                      'Version B: %s\n' % (np.array2string(A1), np.array2string(A2)))
    return A3
