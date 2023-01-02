from scipy.optimize import curve_fit
import numpy as np

from .math_ops import to_rodrigues_vector, to_rotation_matrix, project


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
        P[7 + 6 * i: 7 + 6 * (i + 1)] = w
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
        [    0,     0,  1]
    ])
    k = np.array([k0, k1])
    Ws = []
    M = (P.shape[0] - 7) // 6
    for i in range(M):
        m = 7 + 6 * i
        rho = np.array(P[m:(m+2)+1])
        t = np.array(P[m+3:(m+5)+1]).reshape(-1, 1)
        R = to_rotation_matrix(rho)
        W = np.hstack((R, t))
        Ws.append(W)
    return A, k, Ws


def val(X, *params):
    """
    Value function, invoked by optimize()

    :param X: model points: (X_0, Y_0, Z_0=0, X_1, Y_1, Z_1=0, ..., X_N-1, Y_N-1, Z_N-1=0)
    :param params: parameter vector holding 7+6M elements of the associated camera intrinsics matrix A, the lens distortion coefficients k, and the view transformations W = (W0, W1, ..., W_M-1)
    :return: vector with 2MN values
    """
    P = params
    alpha, beta, gamma, uc, vc, k0, k1 = P[:7]
    A = np.array([
        [alpha, gamma, uc],
        [    0,  beta, vc],
        [    0,     0,  1]
    ])
    k = np.array([k0, k1])

    M = (len(P) - 7) // 6
    # because X has been flattened, the number of points N should be divided by 3
    N = X.shape[0] // 3
    Y = np.zeros((2 * M * N, ))
    l = 0
    for i in range(M):
        m = 7 + 6 * i
        rho = np.array(P[m:(m+2)+1])
        t = np.array(P[m+3:(m+5)+1]).reshape(-1, 1)
        R = to_rotation_matrix(rho)
        W = np.hstack((R, t))
        for j in range(N):
            x_proj, y_proj = project(A, W, X[3*j: 3*(j+1)], k)
            Y[2*l] = x_proj
            Y[2*l+1] = y_proj
            l += 1
    return Y


def refine_all(A, k, W, X, U):
    """
    Overall, non-linear refinement.
    To find the intrinsic and extrinsic parameters that minimize the projection error.
    NOTE: We didn't pass the Jacobian function when optimized using LM algorithm and just do the numeric calculation described in section 3.6.4 in reference [2]. 

    :param A: camera intrinsics
    :param k: lens distortion coefficients
    :param W: extrinsic view parameters
    :param X: the target model points
    :param U: the observered sensor points
    :return: refined estimated A_opt, k_opt, W_opt for the camera intrinsics, distortion parameters, 
             and the camera view parameters, respectively
    """
    P_init = compose_parameter_vector(A, k, W)
    # NOTE: because Wi(3x4) will do dot product to Xj(2x1), we should expand Xj to (X, Y, Z=0) to fit the shape(3x1).
    N = X.shape[0]
    X = np.hstack((X, np.zeros((N, 1))))
    X = X.flatten()
    Y = np.array([u.flatten() for u in U]).flatten()
    P_optimized, _ = curve_fit(f=val, xdata=X, ydata=Y, p0=P_init, jac=None)

    return decompose_parameter_vector(P_optimized)