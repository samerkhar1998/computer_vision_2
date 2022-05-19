import math
import cv2
import numpy as np
from matplotlib import pyplot as plt


# find fundamental matrix
def fundamentalMatrix(pts1, pts2, key):
    if key == 8:
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    else:
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_7POINT)
    return F, mask


# get matching image points
def getImagePts(im1, im2, varName1, varName2, nPoints):
    plt.figure()
    plt.imshow(im1, cmap='gray')
    plt.title("Original Image")
    Pts1 = plt.ginput(nPoints, 0)
    plt.figure()
    plt.imshow(im2, cmap='gray')
    plt.title("Destination")
    Pts2 = plt.ginput(nPoints, 0)

    Pts1 = np.round(Pts1, 0)
    Pts2 = np.round(Pts2, 0)

    imagePts1 = np.ndarray((nPoints, 3), dtype=int)
    imagePts2 = np.ndarray((nPoints, 3), dtype=int)
    for i in range(nPoints):
        imagePts1[i] = np.append(Pts1[i], 1)
        imagePts2[i] = np.append(Pts2[i], 1)
        curr = imagePts1[i][0]
        imagePts1[i][0] = imagePts1[i][1]
        imagePts1[i][1] = curr
        curr = imagePts2[i][0]
        imagePts2[i][0] = imagePts2[i][1]
        imagePts2[i][1] = curr
    np.save(varName1 + ".npy", imagePts1.round())
    np.save(varName2 + ".npy", imagePts2.round())


def computeEpilines(points, whichImage, F, lines=None):
    if whichImage == 1:
        lines = np.matmul(F, points)
        lines = lines.T
    if whichImage == 2:
        F = F.T
        lines = np.matmul(F, points)
        lines = lines.T
    return lines


def convert2Dto3D(points1, points2):
    homo_pts1 = []
    homo_pts2 = []
    for point1, point2 in zip(points1, points2):
        x = point1[0]
        y = point1[1]
        z = 1
        homo_pts1.append((x, y, z))
        x = point2[0]
        y = point2[1]
        z = 1
        homo_pts2.append((x, y, z))
    homo_pts1 = np.int32(homo_pts1)
    homo_pts2 = np.int32(homo_pts2)
    return homo_pts1, homo_pts2


def AlgebraicDistance(F, points1, points2):
    number_of_points = np.size(points2) / 3
    sum = 0
    for point1, point2 in zip(points1, points2):
        sum += abs(np.matmul(np.matmul(point2, F), point1.T))
    sum = sum / number_of_points
    return sum


def EpipolarDistance(F, points1, points2):
    number_of_points = np.size(points1) / 3
    sum = 0
    for point1, point2 in zip(points1, points2):
        l2 = np.matmul(F, point1.T)
        l2 = l2.T
        divisor = math.sqrt(np.square(l2[0]) + np.square(l2[1]))
        d1 = np.square(np.matmul(point2, l2.T) / divisor)
        l1 = np.matmul(F.T, point2.T)
        l1 = l1.T
        divisor = math.sqrt(np.square(l1[0]) + np.square(l1[1]))
        d2 = np.square(np.matmul(point1, l1.T) / divisor)
        sum += d1 + d2
    sum = sum / number_of_points
    return sum


def drawLines(img1, img2, lines1, lines2, pts1, pts2):
    r = img1.shape[0]
    c = img1.shape[1]
    for r1, r2, pt1, pt2 in zip(lines1, lines2, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r1[2] / r1[1]])
        x1, y1 = map(int, [c, -(r1[2] + r1[0] * c) / r1[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 3)
        x0, y0 = map(int, [0, -r2[2] / r2[1]])
        x1, y1 = map(int, [c, -(r2[2] + r2[0] * c) / r2[1]])
        img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 3)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def runScript(pts1, pts2):
    F, mask = fundamentalMatrix(pts1, pts2, 7)
    homo_pts1, homo_pts2 = convert2Dto3D(pts1, pts2)
    lines1 = computeEpilines(homo_pts2.T, 2, F)
    lines2 = computeEpilines(homo_pts1.T, 1, F)
    alg = AlgebraicDistance(F, homo_pts1, homo_pts2)
    img3, img4 = drawLines(img1, img2, lines1, lines2, pts1, pts2)
    epi = EpipolarDistance(F, homo_pts1, homo_pts2)

    plt.subplot(121), plt.imshow(img3)
    plt.title("8 Point algorithm")
    plt.subplot(122), plt.imshow(img4)
    plt.title("Algebraic distance: " + str(round(alg, 4)) + ' Epipolar distance: ' + str(round(epi, 4)))

    plt.show()


if __name__ == "__main__":
    path1 = "Q1/location_1_frame_001.jpg"
    img1 = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)
    path2 = "Q1/location_1_frame_002.jpg"
    img2 = cv2.cvtColor(cv2.imread(path2), cv2.COLOR_BGR2RGB)

    # getImagePts(img1, img2, "loc2_001", "loc2_002", 10)
    pts1 = np.load("loc1_001.npy")
    pts2 = np.load("loc1_002.npy")
    pts1 = [[pt[0], pt[1]] for pt in pts1]
    pts2 = [[pt[0], pt[1]] for pt in pts2]
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    runScript(pts1, pts2)

    path3 = "Q1/location_2_frame_001.jpg"
    img3 = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)
    path4 = "Q1/location_2_frame_002.jpg"
    img4 = cv2.cvtColor(cv2.imread(path2), cv2.COLOR_BGR2RGB)

    # getImagePts(img1, img2, "loc2_001", "loc2_002", 10)
    pts3 = np.load("loc2_001.npy")
    pts4 = np.load("loc2_002.npy")
    pts3 = [[pt[0], pt[1]] for pt in pts3]
    pts4 = [[pt[0], pt[1]] for pt in pts4]
    pts3 = np.int32(pts3)
    pts4 = np.int32(pts4)

    runScript(pts3, pts4)


