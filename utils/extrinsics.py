import numpy as np


def make_true_rotation_matrix(R):
    """
    Calculate the most similar `true` rotation matrix based on SVD.

    :param R: initial rotation matrix
    :return: the most similar `true` rotation matrix
    """
    u, _, vh = np.linalg.svd(R)
    true_R = np.dot(u, vh)
    return true_R


def estimate_view_transform(A, H):
    """
    Estimate a rotation matrix and a translation vector from a single homography matrix

    :param A: intrinsics parameters
    :param H: a homography matrix
    :return: a matrix contains a rotation matrix and a translation vector
    """
    h0 = H[:, 0]
    h1 = H[:, 1]
    h2 = H[:, 2]
    K = 1 / np.linalg.norm(np.dot(np.linalg.inv(A), h0))
    r0 = K * np.dot(np.linalg.inv(A), h0)
    r1 = K * np.dot(np.linalg.inv(A), h1)
    r2 = np.cross(r0, r1)
    t = K * np.dot(np.linalg.inv(A), h2).reshape(-1, 1)
    R_init = np.array([r0, r1, r2]).T
    R = make_true_rotation_matrix(R_init)
    W = np.hstack((R, t))
    return W


def get_extrinsics(A, H):
    """
    Extract extrinsics parameters from homography matrices

    :param A: intrinsics parameters
    :param H: a sequence of homography matrices
    :return: a sequence of extrinsic view parameters W
    """
    Ws = []
    M = len(H)
    for i in range(M):
        W = estimate_view_transform(A, H[i])
        Ws.append(W)
    return Ws