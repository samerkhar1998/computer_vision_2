import random
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib.pyplot as plt
import numpy as np


def readMatricesAndPts():
    cMatrix1 = open("Q2/cameraMatrix1.txt", "r").readlines()
    cMatrix2 = open("Q2/cameraMatrix2.txt", "r").readlines()
    mPts1 = open("Q2/matchedPoints1.txt", "r").readlines()
    mPts2 = open("Q2/matchedPoints2.txt", "r").readlines()
    points1 = []
    points2 = []
    for p1, p2 in zip(mPts1, mPts2):
        p1 = p1.split(",")
        p2 = p2.split(",")
        points1.append([float(p1[0]), float(p1[1])])
        points2.append([float(p2[0]), float(p2[1])])

    mat1 = []
    mat2 = []

    for l1, l2 in zip(cMatrix1, cMatrix2):
        l1 = l1.split(",")
        l2 = l2.split(",")

        mat1.append([float(l1[0]), float(l1[1]), float(l1[2]), float(l1[3])])
        mat2.append([float(l2[0]), float(l2[1]), float(l2[2]), float(l2[3])])

    return np.array(mat1), np.array(mat2), np.array(points1), np.array(points2)


def drawLineBetweenPts(p1, p2, house1, house2):
    ptsLength = len(p1)

    for i in range(ptsLength - 1):
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

        P1 = (int(p1[i][0]), int(p1[i][1]))
        P2 = (int(p1[i + 1][0]), int(p1[i + 1][1]))
        cv2.line(house1, P1, P2, color=color, thickness=3)
        house1 = cv2.circle(house1, tuple(P1), 5, color, -1)
        house1 = cv2.circle(house1, tuple(P2), 5, color, -1)

        P1 = (int(p2[i][0]), int(p2[i][1]))
        P2 = (int(p2[i + 1][0]), int(p2[i + 1][1]))
        cv2.line(house2, P1, P2, color=color, thickness=3)
        house2 = cv2.circle(house2, tuple(P1), 8, color, -1)
        house2 = cv2.circle(house2, tuple(P2), 8, color, -1)

    plt.figure()
    plt.subplot(121), plt.imshow(house1), plt.title("house1")
    plt.subplot(122), plt.imshow(house2), plt.title("house2")
    plt.show()


def DLT(m1, m2, p1, p2):
    '''
    This function has taken from(the link was attached in ex2.pdf):
    https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html#:~:text=def-,DLT,-(P1%2C%20P2%2C%20point1%2C%20point2)%3A%0A%20%0A%20%20%20%20A
    '''
    A = [p1[1] * m1[2, :] - m1[1, :],
         m1[0, :] - p1[0] * m1[2, :],
         p2[1] * m2[2, :] - m2[1, :],
         m2[0, :] - p2[0] * m2[2, :]
         ]
    A = np.array(A).reshape((4, 4))
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices=False)
    return Vh[3, 0:3] / Vh[3, 3]


def new3dPts(m1, m2, p1, p2):
    points = []
    for i in range(len(p1)):
        point = DLT(m1, m2, p1[i], p2[i])
        points.append(point)
    return np.array(points)


def subtractMean(PTS_3D):
    X = 0
    Y = 0
    Z = 0
    length = len(PTS_3D)
    toRet = []
    for x, y, z in PTS_3D:
        X += (x / length)
        Y += (y / length)
        Z += (z / length)
    for i in range(length):
        x = PTS_3D[i][0] - X
        y = PTS_3D[i][1] - Y
        z = PTS_3D[i][2] - Z
        toRet.append((x, y, z))

    return np.array(toRet)


def project_xy(points, title="Matching projection", save=False, axe="x", k=0):
    plt.figure()
    plt.axis([-3, 3, -3, 3])
    for i in range(len(points) - 1):
        plt.plot([points[i][0], points[i + 1][0]], [points[i][1], points[i + 1][1]])

    if not save:
        plt.title(title)
        plt.show()
    else:
        image = "gif_images/" + axe + "_rotated_" + str(k) + ".png"
        plt.savefig(image)


def rotate(Pts, rotation, project=True):
    '''
    This function is taken from:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    '''
    p = Pts.copy()
    r = R.from_euler('zxy', rotation, degrees=True)
    rotation = r.apply(p)
    if project:
        project_xy(rotation, "Random rotation")
    else:
        return rotation


def rotations(Pts):
    for i in range(0, 370, 10):
        rotate_x = rotate(Pts, (i, 0, 0), project=False)
        project_xy(rotate_x, save=True, axe="x", k=i)

        rotate_y = rotate(Pts, (0, i, 0), project=False)
        project_xy(rotate_y, save=True, axe="y", k=i)


if __name__ == "__main__":
    house1 = cv2.imread("Q2/house_1.png")
    house2 = cv2.imread("Q2/house_2.png")
    '''stage 1'''
    matrix1, matrix2, matchedP1, matchedP2 = readMatricesAndPts()
    drawLineBetweenPts(matchedP1, matchedP2, house1, house2)
    '''stage2'''
    # TODO: triangulate
    PTS_3D = new3dPts(matrix1, matrix2, matchedP1, matchedP2)
    # TODO: mean subtraction
    Pts = subtractMean(PTS_3D)
    # TODO: draw the points in a linked way by throwing them into the xy plane
    project_xy(Pts)
    # TODO: Run one random three-dimensional rotation on the dots cloud
    rotate(Pts, (random.randint(0, 360), random.randint(0, 360), random.randint(0, 360)))
    # TODO: Repeat the drawing of step 3 in two loops.
    rotations(Pts)
