import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

from src.camera import Camera
import src.structure
import src.processor
import src.features

import argparse

# Parâmetros
# --image1 "images/box002.jpg" --image2 "images/box001.jpg"

def reconstruction():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image1")
    ap.add_argument("-c", "--image2")
    args = vars(ap.parse_args())

    img1 = cv2.imread(args["image1"])
    img2 = cv2.imread(args["image2"])
    pts1, pts2 = src.features.find_correspondence_points(img1, img2)
    points1 = src.processor.cart2hom(pts1)
    points2 = src.processor.cart2hom(pts2)

    fig, ax = plt.subplots(1, 2)
    ax[0].autoscale_view('tight')
    ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax[0].plot(points1[0], points1[1], 'r.')
    ax[1].autoscale_view('tight')
    ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax[1].plot(points2[0], points2[1], 'r.')
    fig.show()

    height, width, ch = img1.shape
    intrinsic = np.array([  # for dino
        [1280, 0, width / 2],
        [0, 1280, height / 2],
        [0, 0, 1]])

    return points1, points2, intrinsic


points1, points2, intrinsic = reconstruction()

# Calcular a matriz essencial com 2 pontos.
# Primeiro, normalizar os pontos
points1n = np.dot(np.linalg.inv(intrinsic), points1)
points2n = np.dot(np.linalg.inv(intrinsic), points2)
E = src.structure.compute_essential_normalized(points1n, points2n)
print('Matriz essencial:', (-E / E[0][1]))


P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2s = src.structure.compute_P_from_essential(E)

ind = -1
for i, P2 in enumerate(P2s):
    # Encontra os parametros corretos da camera
    d1 = src.structure.reconstruct_one_point(
        points1n[:, 0], points2n[:, 0], P1, P2)


    P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
    d2 = np.dot(P2_homogenous[:3, :4], d1)

    if d1[2] > 0 and d2[2] > 0:
        ind = i

P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
#tripoints3d = structure.reconstruct_points(points1n, points2n, P1, P2)
tripoints3d = src.structure.linear_triangulation(points1n, points2n, P1, P2)

fig = plt.figure()
fig.suptitle('Resultado 3D', fontsize=14)
ax = fig.gca(projection='3d')
ax.plot(tripoints3d[0], tripoints3d[1], tripoints3d[2], 'b.')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=-145, azim=-32)
plt.show()