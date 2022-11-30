from scipy.optimize import curve_fit
import numpy as np


def hom(x):
    """
    Convert an n-dimensional Cartesian point x = (x_0, x_1, ..., x_n-1)^T to homogeneous coordinates.
    We use the notation hom(x) = (x_0, x_1, ..., x_n-1, 1)^T

    :param x: an n-dimensional Cartesian point
    :return: a homogeneous point
    """
    x = np.append(x, 1)
    return x


def hom_inv(x):
    """
    Convert a homogeneous point x = (x_0, x_1, ..., x_n-1, 1)^T to Cartesian coordinate.
    We use the notation hom_inv(x) = (x_0, x_1, ..., x_n-1)^T

    :param x: a homogeneous point
    :return: an n-dimensional Cartesian point
    """
    return x[:-1]


def get_normalization_matrix(X):
    """
    Get Z-score normalization matrix for X = (X_0, X_1, ..., X_N-1), with xj = (xj, yj)^T

    :param X: matrix to be normalized
    :return: normalization matrix
    """
    if X.ndim != 2:
        raise ValueError('X should be 2D dimensions.\n'
                         'Input X dimensions: %d' % X.ndim)

    if X.shape[1] != 2:
        raise ValueError('X should be Nx2 matrix.\n'
                         'Input X shape: (%d, %d)' % X.shape)

    x_mean, y_mean = np.mean(X[:, 0]), np.mean(X[:, 1])
    x_var, y_var = np.var(X[:, 0]), np.var(X[:, 1])
    sx, sy = np.sqrt(2 / x_var), np.sqrt(2 / y_var)
    normalized_X = np.array([
        [sx, 0, -sx * x_mean],
        [0, sy, -sy * y_mean],
        [0,  0,            1]
    ])
    return normalized_X


def solve(M):
    """
    Solve the DLT system using SVD.

    :param M: matrix to be solved
    :return: homogeneous matrix
    """
    _, s, vh = np.linalg.svd(M)
    h = vh[np.argmin(s), :]
    return h


def optimize(val, jac, X, Y, h):
    """
    Use non-linear least squares with LM algorithm to fit a function, val, to data.

    Assumes Y = f(X, *params) + eps.

    :param val: fitting function to data
    :param jac: Jacobian function which is dY/dX
    :param X: model points X: (X_0, Y_0, X_1, Y_1, ..., X_N-1, Y_N-1)
    :param Y: sensor points: (u_0, v_0, u_1, v_1, ..., u_N-1, v_N-1)
    :param h: parameter vector holding 9 elements of the associated homography matrix X: (h0, h1, ..., h8)
    :return: optimized flattened homography matrix
    """
    popt, _ = curve_fit(f=val, xdata=X, ydata=Y, p0=h, method='lm', jac=jac)
    refined_h = popt
    return refined_h


def val(X, *params):
    """
    Value function, invoked by optimize()

    :param X: model points: (X_0, Y_0, X_1, Y_1, ..., X_N-1, Y_N-1)
    :param params: parameter vector holding 9 elements of the associated homography matrix X: (h0, h1, ..., h8)
    :return: vector with 2N values
    """
    # because X has been flattened, the number of points N should be divided by 2
    N = X.shape[0] // 2
    Y = np.zeros(2 * N)
    h = params
    for j in range(N):
        x, y = X[2 * j], X[2 * j + 1]
        w = h[6] * x + h[7] * y + h[8]
        u = (h[0] * x + h[1] * y + h[2]) / w
        v = (h[3] * x + h[4] * y + h[5]) / w
        Y[2 * j] = u
        Y[2 * j + 1] = v
    return Y


def jac(X, *params):
    """
    Jacobian function, invoked by optimize()

    :param X: model points: (X_0, Y_0, X_1, Y_1, ..., X_N-1, Y_N-1)
    :param params: parameter vector holding 9 elements of the associated homography matrix X: (h0, h1, ..., h8)
    :return: Jacobian matrix of size 2N x 9
    """
    # because X has been flattened, the number of points N should be divided by 2
    N = X.shape[0] // 2
    J = np.zeros((2 * N, 9))
    h = params
    for j in range(N):
        x, y = X[2 * j], X[2 * j + 1]
        sx = h[0] * x + h[1] * y + h[2]
        sy = h[3] * x + h[4] * y + h[5]
        w = h[6] * x + h[7] * y + h[8]
        J[2 * j, :] = [x / w, y / w, 1 / w, 0, 0, 0, -sx * x / w**2, -sx * y / w**2, -sx / w**2]
        J[2 * j + 1, :] = [0, 0, 0, x / w, y / w, 1 / w, -sy * x / w**2, -sy * y / w**2, -sy / w**2]
    return J


def v_pq(H, p, q):
    """
    Given homography H, do v_p,q operation defined in eq.96 in reference [2].

    :param H: homography to be operated
    :param p: p column of homography
    :param q: q column of homography
    :return: a 6-dimensional vector
    """
    v_pq = np.array([
        H[0, p] * H[0, q],
        H[0, p] * H[1, q] + H[1, p] * H[0, q],
        H[1, p] * H[1, q],
        H[2, p] * H[0, q] + H[0, p] * H[2, q],
        H[2, p] * H[1, q] + H[1, p] * H[2, q],
        H[2, p] * H[2, q]
    ])
    return v_pq
