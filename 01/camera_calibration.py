import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image
from camera_calibration_func import cal_intrinsic_matrix, compute_homography

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.

#   Regard the provided images as the input.
#   The chessboard size should be like the value below.
# corner_x = 7
# corner_y = 7

#   Regard the images that we take as the input.
#   The chessboard size should be like the value below.
corner_x = 7
corner_y = 10

objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Set the Homography array.
homographies = []

# Make a list of calibration images
# images = glob.glob('my_data/*.jpg')
# images = glob.glob('data/*.jpg')
images = glob.glob('my_data2/*.jpg')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # If found, add object points, image points
    if (ret == True):
        objpoints.append(objp)
        imgpoints.append(corners)

        # 手動計算單應矩陣 H
        H = compute_homography(objp[:, :2], corners.reshape(-1, 2))
        # eval_H, _ = cv2.findHomography(objp[:, :2], corners.reshape(-1, 2))

        # 將單應矩陣 H 加入 homographies 列表
        homographies.append(H)
        print(f'Homography matrix for image {idx} (after manual calculation):\n{H}\n')
        # print(f'eval H:\n{eval_H}\n')

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)


#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
img_size = img[0].shape
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.

#   Obtain the intrinsic matrix by the funciton we define.
_, K = cal_intrinsic_matrix(homographies)

#   Obtain the intrinsic matrix and other values by the built-in function.
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

#   Get the inverse matrix of the intrinsic matrix, K.
iK = np.linalg.inv(K)

#   Set the extrinsic array
extrinsics = []

#   Calculate the extrinsic matrix of the input images.
for H in homographies:
    h1 = H.T[0]
    h2 = H.T[1]
    h3 = H.T[2]

    #   The fomula of the parameters below shows in the slide 79.
    lamda = 1. / np.linalg.norm(iK @ h1)
    r1 = iK @ h1 * lamda
    r2 = iK @ h2 * lamda
    r3 = np.cross(r1, r2)
    t = iK @ h3 * lamda

    #   Append the extrinsic matrix into the array. 
    extrinsics.append(np.vstack((r1, r2, r3, t)).T)
#   Change the list to array
extrinsics = np.asarray(extrinsics)

# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# camera setting
camera_matrix = K
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600

# chess board setting
board_width = 8
board_height = 6
square_size = 1

# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()

#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""