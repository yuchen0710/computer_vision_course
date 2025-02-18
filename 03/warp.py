import numpy as np
import matplotlib.pyplot as plt
import math
import copy

def warp(img1, img2, H, args):
    #   Get the shape of the image1 and image2
    row_img1, col_img1, _ = img1.shape
    row_img2, col_img2, _ = img2.shape

    #   append the coenor to the aray
    corner_img1 = np.array([[0, 0, 1],
                            [col_img1, row_img1, 1]])
    corner_img2 = np.array([[0, 0, 1],
                            [0, row_img2, 1],
                            [col_img2, 0, 1],
                            [col_img2, row_img2, 1]])
    #   Trans the coornate from img1 to img2
    corner_trans = np.dot(H, corner_img2.T).T
    corner_trans = corner_trans / corner_trans[:, [2]]

    corners = np.vstack((corner_img1, corner_trans))

    col_max, col_min = np.int32(corners[:, 0].max()), np.int32(corners[:, 0].min())
    row, col = row_img1, col_max - col_min
    image_merge = np.zeros((row, col, 3), dtype = np.int32)

    #   Put thr inage 1 data into the panoramic image 
    image_merge[:, -col_min : col_img1 - col_min, :] = img1

    H_inv = np.linalg.inv(H)
    for y in range(row):
        for x in range(col):
            coor_img1 = np.array([x, y, 1])
            coor_img2 = np.dot(H_inv, coor_img1)
            coor_img2 /= coor_img2[2]

            if (coor_img2[0] < 0 or coor_img2[0] > (col_img2 - 1) or coor_img2[1] < 0 or coor_img2[1] >= (row_img2 - 1)):
                # image_merge[y, x, :] = np.array([0, 0, 0])
                continue
                
            if np.array_equal(image_merge[y, x], np.array([0, 0, 0])):
                image_merge[y, x, :] = interpolation(img2, coor_img2)
            else:
                ratio = blending_ratio(x, col_img1, args)
                image_merge[y, x, :] = np.int32(image_merge[y, x, :] * ratio + interpolation(img2, coor_img2) * (1 - ratio))

    return image_merge

def interpolation(image, coor):
    row_floor, row_ceil = math.floor(coor[1]), math.ceil(coor[1])
    col_floor, col_ceil = math.floor(coor[0]), math.ceil(coor[0])

    rgb = np.array([image[row_floor, col_floor, :],
                    image[row_floor, col_ceil, :],
                    image[row_ceil, col_floor, :],
                    image[row_ceil, col_ceil, :]])

    return np.int32(np.mean(rgb, 0))

def blending_ratio(x, col_img, args):
    if args.mode == 'linear':
        return min((col_img - x) / col_img, 1)
    elif args.mode == 'sigmoid':
        ratio = 1 / (1 + math.exp(-((col_img - args.blend_threshold) - x) / args.blend_threshold))
        return ratio
    else:
        return 0.4