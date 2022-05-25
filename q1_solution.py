import math
import cv2
import numpy as np
from matplotlib import pyplot as plt


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
    np.save("Q1/" + varName1 + ".npy", imagePts1.round())
    np.save("Q1/" + varName2 + ".npy", imagePts2.round())


def convert2Dto3D(pts1_1, pts1_2, pts2_1, pts2_2):
    homo_pts1 = []
    homo_pts2 = []
    homo_pts3 = []
    homo_pts4 = []
    for p1_1, p1_2, p2_1, p2_2 in zip(pts1_1, pts1_2, pts2_1, pts2_2):
        homo_pts1.append((p1_1[0], p1_1[1], 1))
        homo_pts2.append((p1_2[0], p1_2[1], 1))
        homo_pts3.append((p2_1[0], p2_1[1], 1))
        homo_pts4.append((p2_2[0], p2_2[1], 1))
    homo_pts1 = np.int32(homo_pts1)
    homo_pts2 = np.int32(homo_pts2)
    homo_pts3 = np.int32(homo_pts3)
    homo_pts4 = np.int32(homo_pts4)
    return homo_pts1, homo_pts2, homo_pts3, homo_pts4


def EpipolarDistance(F, points1, points2):
    number_of_points = np.size(points1) / 3
    Sum = 0
    for point1, point2 in zip(points1, points2):
        l2 = np.matmul(F, point1.T)
        l2 = l2.T
        divisor = math.sqrt(np.square(l2[0]) + np.square(l2[1]))
        d1 = np.matmul(point2, l2.T / divisor)
        l1 = np.matmul(F.T, point2.T)
        l1 = l1.T
        divisor = math.sqrt(np.square(l1[0]) + np.square(l1[1]))
        d2 = np.matmul(point1, l1.T / divisor)
        Sum += d1 + d2
    Sum = abs(Sum) / number_of_points
    return Sum


def drawLines(img1, img2, lines1, lines2, pts1, pts2):
    r = img1.shape[0]
    c = img1.shape[1]
    for r1, r2, pt1, pt2 in zip(lines1, lines2, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r1[2] / r1[1]])
        x1, y1 = map(int, [c, -(r1[2] + r1[0] * c) / r1[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 2)
        x0, y0 = map(int, [0, -r2[2] / r2[1]])
        x1, y1 = map(int, [c, -(r2[2] + r2[0] * c) / r2[1]])
        img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 2)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def compute(p1, p2, homo_p1, homo_p2, img1, img2, F):
    left = cv2.computeCorrespondEpilines(p2.reshape(-1, 1, 2), 2, F)
    left = left.reshape(-1, 3)

    right = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2), 1, F)
    right = right.reshape(-1, 3)
    img3, img4 = drawLines(img1, img2, left, right, p1, p2)

    epi = EpipolarDistance(F, homo_p1, homo_p2)

    plt.figure()
    plt.subplot(121), plt.imshow(img3)
    plt.title("SED:" + str(round(epi, 4)))
    plt.subplot(122), plt.imshow(img4)

    plt.show()


def preparation(img1, img2, location):
    # getImagePts(img1, img2, location + "s1_1", location + "s1_2", 10)
    # getImagePts(img1, img2, location + "s2_1", location + "s2_2", 10)

    pts1_1 = np.load("Q1/" + location + "s1_1.npy")
    pts1_2 = np.load("Q1/" + location + "s1_2.npy")
    pts2_1 = np.load("Q1/" + location + "s2_1.npy")
    pts2_2 = np.load("Q1/" + location + "s2_2.npy")

    pts1_1 = [[pt[0], pt[1]] for pt in pts1_1]
    pts1_2 = [[pt[0], pt[1]] for pt in pts1_2]
    pts1_1 = np.int32(pts1_1)
    pts1_2 = np.int32(pts1_2)

    pts2_1 = [[pt[0], pt[1]] for pt in pts2_1]
    pts2_2 = [[pt[0], pt[1]] for pt in pts2_2]
    pts2_1 = np.int32(pts2_1)
    pts2_2 = np.int32(pts2_2)
    # computing fundamental matrix for s1
    F, mask = cv2.findFundamentalMat(pts1_1, pts1_2, cv2.FM_8POINT)
    # for s1 and s2 making 3d points
    homo_pts1_1, homo_pts1_2, homo_pts2_1, homo_pts2_2 = convert2Dto3D(pts1_1, pts1_2, pts2_1, pts2_2)
    return pts1_1, pts1_2, pts2_1, pts2_2, homo_pts1_1, homo_pts1_2, homo_pts2_1, homo_pts2_2, F


if __name__ == "__main__":
    path1 = "Q1/location_1_frame_001.jpg"
    image1 = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)
    path2 = "Q1/location_1_frame_002.jpg"
    image2 = cv2.cvtColor(cv2.imread(path2), cv2.COLOR_BGR2RGB)

    path3 = "Q1/location_2_frame_001.jpg"
    image3 = cv2.cvtColor(cv2.imread(path3), cv2.COLOR_BGR2RGB)
    path4 = "Q1/location_2_frame_002.jpg"
    image4 = cv2.cvtColor(cv2.imread(path4), cv2.COLOR_BGR2RGB)

    '''
    location 1
    '''
    pts1_1, pts1_2, pts2_1, pts2_2, homo_pts1_1, homo_pts1_2, homo_pts2_1, homo_pts2_2, F = preparation(image1, image2,
                                                                                                        "location1_")
    # s1
    one, two = image1.copy(), image2.copy()
    compute(pts1_1, pts1_2, homo_pts1_1, homo_pts1_2, one, two, F)
    # s2
    compute(pts2_1, pts2_2, homo_pts2_1, homo_pts2_2, image1, image2, F)

    '''
    location 2
    '''
    pts1_1, pts1_2, pts2_1, pts2_2, homo_pts1_1, homo_pts1_2, homo_pts2_1, homo_pts2_2, F = preparation(image3, image4,
    "location2_")
    # s1
    three, four = image3.copy(), image4.copy()
    compute(pts1_1, pts1_2, homo_pts1_1, homo_pts1_2, three, four, F)
    # s2
    compute(pts2_1, pts2_2, homo_pts2_1, homo_pts2_2, image3, image4, F)
