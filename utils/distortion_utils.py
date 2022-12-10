from .math_utils import hom, hom_inv, solve

import numpy as np


def estimate_lens_distortion(A: np.ndarray, W: list, X: np.ndarray, U: list):
    """
    Estimation of the radial lens distortion parameters.
    Here we implement four methods to compare the results.

    :param A: the estimated intrinsic parameters
    :param W: the estimated extrinsic parameters; W = (W0, W1, ..., W_M-1) with M views where Wi = (Ri | ti)
    :param X: the target model points; X = (X0, X1, ..., X_N-1) with N points
    :param U: the observed sensor points; Ui = (U_i0, U_i1, ..., U_i,N-1) being the N points for view i
    :return: the list of estimated lens distortion coefficients
    """
    M = len(W)
    N = X.shape[0]
    uc, vc = A[0, 2], A[1, 2]
    D = np.zeros((2 * M * N, 2))
    d = np.zeros((2 * M * N, ))
    l = 0
    # NOTE: because Wi(3x4) will do dot product to Xj(2x1), we should expand Xj to (X, Y, 0) to fit the shape(3x1).
    X = np.hstack((X, np.zeros((N, 1))))
    for i in range(M):
        for j in range(N):
            x, y = hom_inv(np.dot(W[i], hom(X[j, :])))
            r = np.sqrt(x**2 + y**2)
            u, v = np.dot(A[:2, :], hom(hom_inv(np.dot(W[i], hom(X[j, :])))))
            du, dv = u - uc, v - vc
            D[2 * l, :] = [du * r**2, du * r**4]
            D[2 * l + 1, :] = [dv * r**2, dv * r**4]
            u_obs, v_obs = U[i][j, :]
            d[2 * l] = u_obs - u
            d[2 * l + 1] = v_obs - v
            l += 1
    
    # method 1: 
    u, s, vh = np.linalg.svd(D, full_matrices=False)
    D_inv_svd = np.dot(np.dot(vh.T, np.linalg.inv(np.diag(s))), u.T)
    D_inv_pinv = np.linalg.pinv(D)
    assert np.allclose(D_inv_svd, D_inv_pinv), 'Inverse of D computed by SVD and by pinv are different.'
    k1 = np.dot(D_inv_svd, d)

    # method 2: 
    u, s, vh = np.linalg.svd(D, full_matrices=False)
    c = np.dot(u.T, d)
    w = np.linalg.solve(np.diag(s), c)
    k2 = np.dot(vh.T, w)

    # method 3:
    k3 = np.linalg.lstsq(D, d)[0]

    # method 4:
    Q, R = np.linalg.qr(D)
    Qd = np.dot(Q.T, d)
    k4 = np.linalg.solve(R, Qd)

    # Check the lens distortion parameters computed from all methods are the same.
    assert np.allclose(k1, k2), 'Lens distortion solved from 1 and 2 are different.'
    assert np.allclose(k2, k3), 'Lens distortion solved from 2 and 3 are different.'
    assert np.allclose(k3, k4), 'Lens distortion solved from 3 and 4 are different.'

    return k4
