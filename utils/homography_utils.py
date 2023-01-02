import numpy as np

from .math_utils import *


def estimate_homography(P: np.ndarray, Q: np.ndarray):
    """
    Estimate homography matrix using SVD to solve DLT(direct linear transformation).

    :param P: model points in a view
    :param Q: sensor points in the same view as P
    :return: the estimated homography matrix such that qj = H * pj
    """
    if P.shape[0] != Q.shape[0]:
        raise ValueError('Model points P and sensor points Q should be the same amount in the same view.'
                         'P: %d, Q: %d' % (P.shape[0], Q.shape[0]))

    if P.shape[1] != 2 or P.shape[1] != 2:
        raise ValueError('Model points P and sensor points Q must be 2D dimensions.'
                         'p: %d, Q: %d' % (P.shape[1], Q.shape[1]))

    N = P.shape[0]  # number of points in P, Q
    NP = get_normalization_matrix(P)
    NQ = get_normalization_matrix(Q)
    M = np.zeros((2 * N, 9))
    for j in range(N):
        k = 2 * j
        # Normalize
        norm_p = hom_inv(np.dot(NP, hom(P[j])))
        norm_q = hom_inv(np.dot(NQ, hom(Q[j])))
        M[k, :] = [norm_p[0], norm_p[1], 1, 0, 0, 0, -norm_p[0] * norm_q[0], -norm_p[1] * norm_q[0], -norm_q[0]]
        M[k+1, :] = [0, 0, 0, norm_p[0], norm_p[1], 1, -norm_p[0] * norm_q[1], -norm_p[1] * norm_q[1], -norm_q[1]]

    h = solve(M)
    normalized_H = h.reshape(3, 3)
    # de-normalize
    H = np.dot(np.dot(np.linalg.inv(NQ), normalized_H), NP)
    return H


def refine_homography(H, X, U):
    """
    Refinement of a single view homography by minimizing the projection error in the sensor image
    using non-linear(Levenberg-Marquardt) optimization.

    :param H: the initial 3x3 homography matrix
    :param X: model points for a single view
    :param U: sensor points for the same single view as model points
    :return: optimized homography matrix
    """
    X = X.flatten()
    Y = U.flatten()
    h = H.flatten()
    h_optimized = optimize(val, X, Y, h, jac)
    H_optimized = h_optimized.reshape(3, 3) / h_optimized[8]
    return H_optimized


def get_homographies(X: np.ndarray, U: list):
    """
    Given model points and the associate sensor points in M views, get initial homographies.

    :param X: model points
    :param U: sensor points
    :return: a sequence of estimated homographies Hs = (H_0, H_1, ..., H_M-1), one for each of the M views.
    """

    # number of views
    M = len(U)
    Hs = []
    for i in range(M):
        H_init = estimate_homography(X, U[i])
        H = refine_homography(H_init, X, U[i])
        Hs.append(H)
    return Hs


