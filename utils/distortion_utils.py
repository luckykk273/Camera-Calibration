from .math_utils import hom, hom_inv, solve

import numpy as np


def estimate_lens_distortion(A, W, X, U):
    """
    Estimation of the radial lens distortion parameters.

    :param A: the estimated intrinsic parameters
    :param W: the estimated extrinsic parameters; W = (W0, W1, ..., W_M-1) with M views where Wi = (Ri | ti)
    :param X: the target model points; X = (X0, X1, ..., X_N-1) with N points
    :param U: the observed sensor points; Ui = (U_i0, U_i1, ..., U_i,N-1) being the N points for view i
    :return: the list of estimated lens distortion coefficients
    """
    M = len(W)
    N = len(X)
    uc, vc = A[0, 2], A[1, 2]
    D = np.zeros((2 * M * N, 2))
    d = np.zeros((2 * M * N, ))
    l = 0
    for i in range(M):
        for _ in range(N):
            x, y = hom_inv(np.dot(W[i], hom(X[i])))
            r = np.sqrt(x**2 + y**2)
            u, v = np.dot(A[:2, :], hom(hom_inv(np.dot(W[i], hom(X[i])))))
            du, dv = u - uc, v - vc
            D[2 * l, :] = [du * r**2, du * r**4]
            D[2 * l + 1, :] = [dv * v**2, dv * v**4]
            u_obs, v_obs = U[i]
            d[2 * l] = u_obs - u
            d[2 * l + 1] = v_obs - v
            l += 1
    # TODO:
    # k = solve(D * k = d) linear least squares solution, e.g., by SVD
    return k
