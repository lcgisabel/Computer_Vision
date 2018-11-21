import cv2
import numpy as np

img = cv2.imread('img/floor.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Armazenar as coordenadas para as linhas

list1 = []
list2 = []
check = 0

def checkPoints():
    while True:
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("a"):
            break


def getPoint(event, x, y, flags, params):
    global list1, list2, check

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        if check < 4:
            list1.append([x, y])
            check += 1
        else:
            list2.append([x, y])
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
        cv2.imshow("Original Image", img)

cv2.setMouseCallback("image", getPoint)

checkPoints();

cv2.destroyAllWindows()
cv2.waitKey(1)


#===================================== Retificação Afim ===================================================

# Equação da linha de fuga
def vanishingLine(list1, list2):
    l1 = np.cross([list1[0][0], list1[0][1], 1], [list1[1][0], list1[1][1], 1])
    l2 = np.cross([list1[2][0], list1[2][1], 1], [list1[3][0], list1[3][1], 1])
    p1 = np.cross([list2[0][0], list2[0][1], 1], [list2[1][0], list2[1][1], 1])
    p2 = np.cross([list2[2][0], list2[2][1], 1], [list2[3][0], list2[3][1], 1])

    temp1 = np.cross(l1, l2)
    temp2 = np.cross(p1, p2)

    v1 = temp1 / temp1[2]
    v2 = temp2 / temp2[2]

    return np.cross(v1, v2)


vLine = vanishingLine(list1, list2)


# Retorna H da retificação Afim
def affineHom(line):
    return np.array([[1, 0, 0], [0, 1, 0], [line[0] / line[2], line[1] / line[2], 1]])


HomMatrix = np.float32(affineHom(vLine))

size = gray.shape
sizeNew = (size[1], size[0])
AffineRect = cv2.warpPerspective(img, HomMatrix, sizeNew)
cv2.imwrite("result/Affine.jpg", AffineRect)
cv2.waitKey(500)

#===================================== Retificação Métrica ===================================================

img = AffineRect
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Listas com pontos para o cálculo das linhas perpendiculares
Mlist1 = []
Mlist2 = []
checkM = 0


def getPointMetric(event, x, y, flags, params):
    global Mlist1, Mlist2, checkM

    if event == cv2.EVENT_LBUTTONDOWN:
        print (x, y)
        if checkM < 3:
            Mlist1.append([x, y])
            checkM += 1
        else:
            Mlist2.append([x, y])
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
        cv2.imshow("Affine Image", img)

cv2.setMouseCallback("image", getPointMetric)

checkPoints()
cv2.destroyAllWindows()
cv2.waitKey(1)


# Retorna as linhas perperndiculares para a retificação Métrica
def returnPerLines(list1, list2):

    l1 = np.cross([list1[0][0], list1[0][1], 1], [list1[1][0], list1[1][1], 1])
    m1 = np.cross([list1[2][0], list1[2][1], 1], [list1[1][0], list1[1][1], 1])

    l2 = np.cross([list2[0][0], list2[0][1], 1], [list2[1][0], list2[1][1], 1])
    m2 = np.cross([list2[2][0], list2[2][1], 1], [list2[1][0], list2[1][1], 1])

    return l1 / l1[2], m1 / m1[2], l2 / l2[2], m2 / m2[2]


l1, m1, l2, m2 = returnPerLines(Mlist1, Mlist2)


# Retorna a matriz simétrica
def symmetricMatrix(l1, m1, l2, m2):
    C = np.array([-l1[1] * m1[1], -l2[1] * m2[1]])
    A = np.array([[l1[0] * m1[0], l1[0] * m1[1] + l1[1] * m1[0]], [l2[0] * m2[0], l2[0] * m2[1] + l2[1] * m2[0]]])

    s = np.matmul(np.linalg.inv(A), C)

    sWhole = np.array([[s[0], s[1]], [s[1], 1]])
    return sWhole


smatrix = symmetricMatrix(l1, m1, l2, m2)

U, D, V = np.linalg.svd(smatrix)


# Retorna a matriz homografica afim
def affineHom(U, D, V):
    Dtemp = np.sqrt(D)
    Dfinal = np.array([[Dtemp[0], 0], [0, Dtemp[1]]])

    Am = np.matmul(np.matmul(U, Dfinal), V)

    return np.array([[Am[0][0], Am[0][1], 0], [Am[1][0], Am[1][1], 0], [0, 0, 1]])


HomAffine = affineHom(U, D, V)
InvHomeAffine = np.float32(np.linalg.inv(HomAffine))

MetricRect = cv2.warpPerspective(AffineRect, InvHomeAffine, sizeNew)

cv2.imshow("Metric Rectification", MetricRect)
cv2.imwrite("result/Metric", MetricRect)
