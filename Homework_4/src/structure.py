import matplotlib.pyplot as plt
import numpy as np

def reconstruct_points(p1, p2, m1, m2):
    num_points = p1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        res[:, i] = reconstruct_one_point(p1[:, i], p2[:, i], m1, m2)

    return res


def reconstruct_one_point(pt1, pt2, m1, m2):
    A = np.vstack([
        np.dot(skew(pt1), m1),
        np.dot(skew(pt2), m2)
    ])
    U, S, V = np.linalg.svd(A)
    P = np.ravel(V[-1, :4])

    return P / P[3]


def linear_triangulation(p1, p2, m1, m2):
    num_points = p1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        A = np.asarray([
            (p1[0, i] * m1[2, :] - m1[0, :]),
            (p1[1, i] * m1[2, :] - m1[1, :]),
            (p2[0, i] * m2[2, :] - m2[0, :]),
            (p2[1, i] * m2[2, :] - m2[1, :])
        ])

        _, _, V = np.linalg.svd(A)
        X = V[-1, :4]
        res[:, i] = X / X[3]

    return res


def compute_epipole(F):

    # return(Fx=0)
    U, S, V = np.linalg.svd(F)
    e = V[-1]
    return e / e[2]


def plot_epipolar_lines(p1, p2, F, show_epipole=False):

    plt.figure()
    plt.suptitle('Epipolar lines', fontsize=16)

    plt.subplot(1, 2, 1, aspect='equal')
    # L1 = F * p2
    plot_epipolar_line(p1, p2, F, show_epipole)
    plt.subplot(1, 2, 2, aspect='equal')
    # L2 = F' * p1
    plot_epipolar_line(p2, p1, F.T, show_epipole)


def plot_epipolar_line(p1, p2, F, show_epipole=False):
    lines = np.dot(F, p2)
    pad = np.ptp(p1, 1) * 0.01
    mins = np.min(p1, 1)
    maxes = np.max(p1, 1)

    xpts = np.linspace(mins[0] - pad[0], maxes[0] + pad[0], 100)
    for line in lines.T:
        ypts = np.asarray([(line[2] + line[0] * p) / (-line[1]) for p in xpts])
        valid_idx = ((ypts >= mins[1] - pad[1]) & (ypts <= maxes[1] + pad[1]))
        plt.plot(xpts[valid_idx], ypts[valid_idx], linewidth=1)
        plt.plot(p1[0], p1[1], 'ro')

    if show_epipole:
        epipole = compute_epipole(F)
        plt.plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')


def skew(x):
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])


def compute_P(p2d, p3d):
    n = p2d.shape[1]
    if p3d.shape[1] != n:
        raise ValueError('sem match.')

    # create matrix for DLT
    M = np.zeros((3 * n, 12 + n))
    for i in range(n):
        M[3 * i, 0:4] = p3d[:, i]
        M[3 * i + 1, 4:8] = p3d[:, i]
        M[3 * i + 2, 8:12] = p3d[:, i]
        M[3 * i:3 * i + 3, i + 12] = -p2d[:, i]

    U, S, V = np.linalg.svd(M)
    return V[-1, :12].reshape((3, 4))


def compute_P_from_fundamental(F):

    #Computar a matriz fundamental assumindo P1 = [I 0]
    e = compute_epipole(F.T)
    Te = skew(e)
    return np.vstack((np.dot(Te, F.T).T, e)).T


def compute_P_from_essential(E):
    U, S, V = np.linalg.svd(E)

    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    P2s = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

    return P2s


def correspondence_matrix(p1, p2):
    p1x, p1y = p1[:2]
    p2x, p2y = p2[:2]

    return np.array([
        p1x * p2x, p1x * p2y, p1x,
        p1y * p2x, p1y * p2y, p1y,
        p2x, p2y, np.ones(len(p1x))
    ]).T

    return np.array([
        p2x * p1x, p2x * p1y, p2x,
        p2y * p1x, p2y * p1y, p2y,
        p1x, p1y, np.ones(len(p1x))
    ]).T


def compute_image_to_image_matrix(x1, x2, compute_essential=False):
    A = correspondence_matrix(x1, x2)

    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)


    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    if compute_essential:
        S = [1, 1, 0]
    F = np.dot(U, np.dot(np.diag(S), V))

    return F


def scale_and_translate_points(points):

    x = points[0]
    y = points[1]
    center = points.mean(axis=1)
    cx = x - center[0]
    cy = y - center[1]
    dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
    scale = np.sqrt(2) / dist.mean()
    norm3d = np.array([
        [scale, 0, -scale * center[0]],
        [0, scale, -scale * center[1]],
        [0, 0, 1]
    ])

    return np.dot(norm3d, points), norm3d


def compute_normalized_image_to_image_matrix(p1, p2, compute_essential=False):

    n = p1.shape[1]
    if p2.shape[1] != n:
        raise ValueError('Sem match.')

    p1n, T1 = scale_and_translate_points(p1)
    p2n, T2 = scale_and_translate_points(p2)

    F = compute_image_to_image_matrix(p1n, p2n, compute_essential)

    # Processar as coordenadas => P1' E P2 = 0
    F = np.dot(T1.T, np.dot(F, T2))

    return F / F[2, 2]


def compute_fundamental_normalized(p1, p2, p3, p4):
    return compute_normalized_image_to_image_matrix(p1, p2, p3, p4)


def compute_essential_normalized(p1, p2):
    return compute_normalized_image_to_image_matrix(p1, p2, compute_essential=True)
