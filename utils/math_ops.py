import cv2
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
    We use the notation hom_inv(x) = (x_0, x_1, ..., x_n-1)^T.
    NOTE: We should devide the last element when converting a homogeneous vector back to Cartesian coordinates 
          assuming that the last element not equal to zero.
          For more detail, please refer to the equation (153) in the appendix A.1 in reference [2]. 

    :param x: a homogeneous point
    :return: an n-dimensional Cartesian point
    """
    return x[:-1] / x[-1]


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


def to_rodrigues_vector(R):
    """
    Transform a 3D rotation matrix to a Rodrigues rotation vector.

    :param R: a 3D rotation matrix
    :return: the associated Rodrigues rotation vector rho
    """
    p = 0.5 * np.array([
        R[2, 1] - R[1, 2], 
        R[0, 2] - R[2, 0], 
        R[1, 0] - R[0, 1]
    ])
    c = 0.5 * (np.trace(R) - 1)
    if np.linalg.norm(p) == 0:
        if c == 1:
            rho1 = np.zeros((3, ))
        elif c == -1:
            R_plus = R + np.identity(3)
            # get column vector of R+ with max. norm
            v = R_plus[:, np.argmax(np.linalg.norm(R_plus, axis=0))]
            u = v / np.linalg.norm(v)
            if (u[0] < 0) or (u[0] == 0 and u[1] < 0) or (u[0] == u[1] == 0 and u[2] < 0):
                u = -u
            rho1 = np.pi * u
        else:
            raise ValueError('The 3D rotation matrix transformed is invalid.')
    else:
        u = p / np.linalg.norm(p)
        theta = np.arctan2(np.linalg.norm(p), c)
        rho1 = theta * u

    rho2 = cv2.Rodrigues(R)[0].reshape((-1, ))
    assert np.allclose(rho1, rho2), 'Transformation from rotation matrix to Rodrigues vector computed from scratch is different from cv2.Rodrigues.'
    return rho1


def to_rotation_matrix(rho):
    """
    Transform  a Rodrigues rotation vector to a 3D rotation matrix.

    :param rho: a Rodrigues rotation vector
    :return: the associated 3D rotation matrix R
    """
    theta = np.linalg.norm(rho)
    rho_hat = rho / np.linalg.norm(rho)
    W = np.array([
        [0, -rho_hat[2], rho_hat[1]],
        [rho_hat[2], 0, -rho_hat[0]],
        [-rho_hat[1], rho_hat[0], 0]
    ])
    R1 = np.identity(3) + W * np.sin(theta) + np.matmul(W, W) * (1 - np.cos(theta))

    R2 = cv2.Rodrigues(rho)[0]
    assert np.allclose(R1, R2), 'Transformation from Rodrigues vector to rotation matrix computed from scratch is different from cv2.Rodrigues.'
    return R1


def warp(x, k):
    """
    Map an undistored 2D coordinate to a distorted 2D coordinate.

    :param x: the undistorted points on the normalized coordinates
    :param k: lens distortion coefficients
    :return: the distorted points on the normalized coordinates
    """
    r = np.linalg.norm(x)
    D = k[0] * r**2 + k[1] * r**4
    return x * (1 + D)


def project(A, W, X, k=None):
    """
    Projection function which maps the 3D point X = (X, Y, Z)(defined in world coordinates) to the 2D sensor point u = (u, v), 
    using the intrinsic parameters A, the extrinsic parameters W and the lens distortion k(optional).
    This function is defined as the equation (24) in the reference [2].

    :param A: camera intrinsics
    :param W: extrinsic view parameters
    :param X: the target model points
    :param k: lens distortion coefficients(optional)
    :return: the projected 2D sensor point u = (u, v)
    """
    x = hom_inv(np.dot(W, hom(X)))
    if k is not None:
        x = warp(x, k)
    u = np.dot(A[:2, :], hom(x))
    return u