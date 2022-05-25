import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from scipy import linalg
from scipy.spatial.transform import Rotation
import os

def Read_cameraMatrix():
    '''this function is part of stage 1, first step, reading data from disk'''
    myfile1 = "Q2/cameraMatrix1.txt"
    with open(myfile1, "r+", encoding='utf-8') as points1:
        points = points1.readlines()
    my_points = []
    for point in points:

        new_point = point.split(",")
        if new_point[-1][-1] == '\n':
            new_point[-1] = new_point[-1][:-1]
        my_points.append(new_point)
    matrix1 = np.array(my_points, dtype=float)
    myfile2 = "Q2/cameraMatrix2.txt"
    with open(myfile2, "r+", encoding='utf-8') as points2:
        points2 = points2.readlines()
    my_points2 = []
    for point in points2:
        new_point = point.split(",")
        if new_point[-1][-1] == '\n':
            new_point[-1] = new_point[-1][:-1]
        my_points2.append(new_point)
    matrix2 = np.array(my_points2, dtype=float)

    return matrix1, matrix2

def Read_matchedpoints():
    '''this function is part of stage 1, first step, reading data from disk'''
    myfile1 = "Q2/matchedPoints1.txt"
    with open(myfile1, "r+", encoding='utf-8') as points1:
        points = points1.readlines()
    my_points1 = []
    for point in points:
        float_point = []
        new_point = point.split(",")
        if new_point[-1][-1] == '\n':
            new_point[-1] = new_point[-1][:-1]
        float_point.append(float(new_point[0]))
        float_point.append(float(new_point[1]))
        my_points1.append(float_point)
    myfile2 = "Q2/matchedPoints2.txt"
    with open(myfile2, "r+", encoding='utf-8') as points2:
        points2 = points2.readlines()
        my_points2 = []
    for point in points2:
        float_point = []
        new_point = point.split(",")
        if new_point[-1][-1] == '\n':
            new_point[-1] = new_point[-1][:-1]
        float_point.append(float(new_point[0]))
        float_point.append(float(new_point[1]))
        my_points2.append(float_point)

    return my_points1, my_points2
def random_Rotations(the_points):
    p = the_points.copy()
    rotation_3d = (random.randint(0, 360), random.randint(0, 360), random.randint(0, 360))
    #print(rotation_3d)
    r = Rotation.from_euler('xyz', rotation_3d, degrees=True)
    p_r = r.apply(p)
    return p_r

def DLT(P1, P2, point1, point2):
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]
         ]

    A = np.array(A).reshape((4,4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)

    return Vh[3, 0:3] / Vh[3, 3]

def x_Rotation(myPoints,num):
    p = myPoints.copy()
    rotation_3d = (num, 0, 0)
    r = Rotation.from_euler('xyz', rotation_3d, degrees=True)
    p_r = r.apply(p)
    return p_r

def y_Rotation(myPoints, num):
    p = myPoints.copy()
    rotation_3d = (0, num, 0)
    r = Rotation.from_euler('xyz', rotation_3d, degrees=True)
    p_r = r.apply(p)
    return p_r
def create_dir(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)
    print("Created Directory:", dir)
    print("generating the 74 images in order to make the gif")
  else:
    print("Directory already exists:", dir)
    print("now generating the 74 images in order to make the gif")
  return dir

if __name__ == '__main__':

    points1, points2 = Read_matchedpoints()
    matrix1, matrix2 = Read_cameraMatrix()
    my_house1 = cv2.imread('Q2/house_1.png')
    my_house2 = cv2.imread('Q2/house_2.png')
    before_last = len(points1)-1
    '''stage 1 : STEP 2 , drawing the lines between the points'''
    for j in range(before_last):
        X11 = int(np.ceil(points1[j][0]))
        Y11 = int(np.ceil(points1[j][1]))
        Pt11 = (X11, Y11)
        X12 = int(np.ceil(points1[j+1][0]))
        Y12 = int(np.ceil(points1[j+1][1]))
        Pt12 = (X12, Y12)
        X21 = int(np.ceil(points2[j][0]))
        Y21 = int(np.ceil(points2[j][1]))
        Pt21 = (X21, Y21)
        X22 = int(np.ceil(points2[j + 1][0]))
        Y22 = int(np.ceil(points2[j + 1][1]))
        Pt22 = (X22, Y22)
        rgb = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        THICKNESS = 4
        cv2.line(my_house1, Pt11, Pt12, color=rgb, thickness=THICKNESS)
        cv2.line(my_house2, Pt21, Pt22, color=rgb, thickness=THICKNESS)

    cv2.imshow("house_1", my_house1)
    cv2.imshow("house_2", my_house2)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # Here we're done with stage 1
    arr_points1 = np.array(points1)
    arr_points2 = np.array(points2)
    '''step 1 triangulating the points from 2D to 3D using a given function DLT'''
    new_3d_points = []
    for j in range(len(points1)):
        new_point = DLT(matrix1, matrix2, arr_points1[j], arr_points2[j])
        new_3d_points.append(new_point)
    new_3d_points = np.array(new_3d_points)
    xy_projected_list = []
    sum_x = 0
    sum_y = 0
    sum_z = 0
    '''step 2 Here we subtract the mean of the points from each coordinate'''
    for j in range(len(points1)):
        sum_x += new_3d_points[j][0]
        sum_y += new_3d_points[j][1]
        sum_z += new_3d_points[j][2]
    avg_x = (sum_x / len(points1))
    avg_y = (sum_y / len(points1))
    avg_z = (sum_z / len(points1))

    for j in range(len(new_3d_points)):
        new_x = new_3d_points[j][0] - avg_x
        new_y = new_3d_points[j][1] - avg_y
        new_z = new_3d_points[j][2] - avg_z
        new_point = (new_x, new_y, new_z)
        xy_projected_list.append(new_point)

    read_data = np.array(xy_projected_list)
    '''step 3 drawing the points on graph and connecting them, (matches xy projected)'''
    my_axes_graph = plt.figure()
    graph_ex = my_axes_graph.add_subplot(111)
    graph_ex.axis([-4, 4, -4, 4])
    for i in range(len(read_data)-1):
        pt1 = xy_projected_list[i]
        pt2 = xy_projected_list[i+1]
        y_values = [(pt1[1] ), (pt2[1])]
        x_values = [(pt1[0]), (pt2[0])]
        graph_ex.plot(x_values, y_values)
    # plt.savefig('Q2/plot.png', dpi=140, bbox_inches='tight')
    graph_ex.set_title("matches_xy_projection")
    plt.show()

    '''step 4 a random 3D rotation on the 3D points I built in the previous step'''
    Random_rotate = random_Rotations(read_data)
    my_axes1_graph = plt.figure()
    graph_ex2 = my_axes1_graph.add_subplot(111)
    graph_ex2.axis([-4, 4, -4, 4])
    for i in range(len(Random_rotate) - 1):
        pt1 = Random_rotate[i]
        pt2 = Random_rotate[i + 1]
        y_values = [(pt1[1]), (pt2[1])]
        x_values = [(pt1[0]), (pt2[0])]
        graph_ex2.plot(x_values, y_values)
    graph_ex2.set_title("Step 4: a random 3D rotation")
    plt.show()

    '''step 5 creating the gif after saving sequence of rotations'''
    plt.rcParams.update({'figure.max_open_warning': 0})
    TEN = 10
    create_dir("gif_images")
    for j in range(37):
        num = TEN * j
        p_r = x_Rotation(Random_rotate, num)
        my_axes1_graph = plt.figure()
        graph_ex2 = my_axes1_graph.add_subplot(111)
        graph_ex2.axis([-4, 4, -4, 4])
        for i in range(len(p_r) - 1):
            pt1 = p_r[i]
            pt2 = p_r[i + 1]
            y_values = [(pt1[1]), (pt2[1])]
            x_values = [(pt1[0]), (pt2[0])]
            graph_ex2.plot(x_values, y_values)
        image = "gif_images/x_rotated_" + str(j) + ".png"
        plt.savefig(image)
        # plt.show()
    for k in range(37):
        num = TEN * k
        p_r = y_Rotation(Random_rotate, num)
        my_axes1_graph = plt.figure()
        graph_ex2 = my_axes1_graph.add_subplot(111)
        graph_ex2.axis([-4, 4, -4, 4])
        for i in range(len(p_r) - 1):
            pt1 = p_r[i]
            pt2 = p_r[i + 1]
            y_values = [(pt1[1]), (pt2[1])]
            x_values = [(pt1[0]), (pt2[0])]
            graph_ex2.plot(x_values, y_values)
        image = "gif_images/y_rotated_" + str(k) + ".png"
        plt.savefig(image)
        # plt.show()

