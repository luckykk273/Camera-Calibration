from .math_utils import to_rodrigues_vector, to_rotation_matrix
import numpy as np


def compose_parameter_vector(A, k, W):
    """
    Compose parameters to be optimized to a parameter vector.

    :param A: camera intrinsics
    :param k: lens distortion coefficients
    :param W: extrinsic view parameters
    :return: a parameter vector P of length 7+6M
    """
    M = len(W)
    P = np.zeros((7 + 6 * M, ))
    a = np.array([A[0, 0], A[1, 1], A[0, 1], A[0, 2], A[1, 2], k[0], k[1]])
    P[:7] = a
    for i in range(M):
        R, t = W[i][:, :3], W[i][:, 3]
        rho = to_rodrigues_vector(R)
        w = np.array([rho[0], rho[1], rho[2], t[0], t[1], t[2]])
        P[7 + 6 * i: 7 + 6 * i + 6] = w
    return P


def decompose_parameter_vector(P):
    """
    Decompose a parameter vector to associated parameters.

    :param P: parameter vector of length 7+6*M(with M being the number of views)
    :return: the associated camera intrinsics matrix A, the lens distortion coefficients k, and the view transformations W = (W0, W1, ..., W_M-1)
    """
    alpha, beta, gamma, uc, vc, k0, k1 = P[:7]
    A = np.array([
        [alpha, gamma, uc],
        [    0,  beta, vc],
        [    0,     1,  1]
    ])
    k = np.array([k0, k1])
    Ws = []
    M = (P.shape[0] - 7) // 6
    for i in range(M):
        m = 7 + 6 * i
        rho = np.array(P[m:(m+2)+1])
        t = np.array(P[m+3:(m+5)+1])
        R = to_rotation_matrix(rho)
        W = np.hstack((R, t))
        Ws.append(W)
    return A, k, Ws


def refine_all(A, k, W, X, U):
    """
    Overall, non-linear refinement.
    To find the intrinsic and extrinsic parameters that minimize the projection error.

    :param A: camera intrinsics
    :param k: lens distortion coefficients
    :param W: extrinsic view parameters
    :param X: the target model points
    :param U: the observered sensor points
    :return: refined estimated A_opt, k_opt, W_opt for the camera intrinsics, distortion parameters, 
             and the camera view parameters, respectively
    """
    P_init = compose_parameter_vector(A, k, W)
    X = X.flatten()
    Y = U.flatten()
    P_optimized = optimize()

    return decompose_parameter_vector(P_init)