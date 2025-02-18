import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def sift(img1, img2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    return keypoints1, descriptors1, keypoints2, descriptors2

def find_matching(descriptors1, descriptors2):
    # Using FLANN matcher
    index_params = dict(algorithm=1, trees=10)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    return matches

def find_correspondences(img1, img2):
    keypoints1, descriptors1, keypoints2, descriptors2 = sift(img1, img2)
    matches = find_matching(descriptors1, descriptors2)   

    good_matches = []
    pts1 = []
    pts2 = []

    # Ratio test as per Lowe's paper
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good_matches.append(m)
            pts1.append(keypoints1[m.queryIdx].pt)
            pts2.append(keypoints2[m.trainIdx].pt)

    return np.array(pts1), np.array(pts2), good_matches

#   Choose 8 points randomly
def get_sample_points(pts1, pts2, points_num = 8):
    #   Check the length of the points
    if (len(pts1) <= points_num) and (len(pts2) <= points_num):
        return pts1, pts2
    
    rng = np.random.default_rng()
    index = rng.choice(np.arange(len(pts1)), size = points_num, replace = False)
    pts1_sample = pts1[index, :]
    pts2_sample = pts2[index, :]

    return pts1_sample, pts2_sample

def normalzation2dpts(points):
    mean = np.mean(points, axis = 0)
    meandist = np.mean(np.linalg.norm((points - mean), axis = 1))
    scale = np.sqrt(2) / meandist

    T = np.array([
        [scale,     0, - scale * mean[0]],
        [    0, scale, - scale * mean[1]],
        [    0,     0,                 1]
    ])

    points_extend = np.hstack((points, np.ones((points.shape[0], 1))))
    points_normalize = (T @ points_extend.T).T
    return points_normalize, T

def get_fundamental_matrix(pts1, pts2):
    pts1_norm, T1 = normalzation2dpts(pts1)
    pts2_norm, T2 = normalzation2dpts(pts2)

    A = np.zeros((len(pts1), 9))
    for index in range(A.shape[0]):
        x1, y1, _ = pts1_norm[index]
        x2, y2, _ = pts2_norm[index]
        
        A[index] = [x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1]

    _, _, V_t = np.linalg.svd(A)
    f = V_t[-1, :]
    F = f.reshape(3, 3)

    U, S, V_t = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ V_t

    #   De-normalize: F = T'^T * F_hat * T
    F = T2.T @ F @ T1
    F /= F[2, 2]

    return F

#   Calculate the projection error
def calculate_error(pts1, pts2, F):
    pts1_extend = np.hstack((pts1, np.ones((len(pts1), 1))))
    pts2_extend = np.hstack((pts2, np.ones((len(pts2), 1))))

    pts1_hat = F @ pts1_extend.T
    pts2_hat = F.T @ pts2_extend.T

    J = pts1_hat[0] ** 2 + pts1_hat[1] ** 2 + pts2_hat[0] ** 2 + pts2_hat[1] ** 2
    J = J.reshape(-1, 1)
    error = np.diag(pts2_extend @ F @ pts1_extend.T) ** 2
    error = error.reshape(-1, 1) / J

    return error

#   Choose the best fundamental matrix by calculating the number of inlier and compare
#   Using 8 points algorithm to find the fundamental matrix
def RANSAC(pts1, pts2, iter_num = 10000, error_threshold = 1.0, confidence = 0.9):
    #   Set the initial value
    matchesLength = len(pts1)
    F_best = []
    pts1_best = None
    pts2_best = None
    maxNum_inlier = 0

    for _ in range(iter_num):
        pts1_sample, pts2_sample = get_sample_points(pts1, pts2)
        F = get_fundamental_matrix(pts1_sample, pts2_sample)

        #   Calculate the projection error
        error = calculate_error(pts1, pts2, F)
        # Obtain the index mask
        index_inlier = error[:, 0] < error_threshold
        num_inlier = index_inlier.astype(np.int32).sum()

        if (num_inlier > maxNum_inlier) and (num_inlier >= matchesLength * confidence):
            F_best = F
            pts1_best = pts1[index_inlier]
            pts2_best = pts2[index_inlier]

    return F_best, pts1_best, pts2_best

def compute_epipolar_line(F, pts1, pts2):
    #   Extend the coordinate
    pts1_extend = np.hstack((pts1, np.ones((len(pts1), 1))))
    pts2_extend = np.hstack((pts2, np.ones((len(pts2), 1))))

    lines1 = (F.T @ pts1_extend.T).T
    lines2 = (F @ pts2_extend.T).T
    n1 = np.sqrt(np.sum(lines1[:, :2] ** 2, axis = 1)).reshape(-1, 1)
    n2 = np.sqrt(np.sum(lines2[:, :2] ** 2, axis = 1)).reshape(-1, 1)

    lines1 = lines1 / n1 * -1
    lines2 = lines2 / n2 * -1

    return lines1, lines2

def get_lines(img, lines, points):
    """ Draws epipolar lines on the given image. """
    r, c, _ = img.shape
    
    for r, pt in zip(lines, points.astype(np.int32)):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img_color = cv2.line(img, (x0, y0), (x1, y1), color, 1)
        img_color = cv2.circle(img, tuple(pt), 5, color, -1)

    return img_color

def draw_epipolar_lines(key, img1_with_lines, img2_with_lines):
    plt.figure(figsize=(15, 10))
    plt.subplot(121), plt.imshow(img1_with_lines)
    plt.title("Epipolar Lines (Image 1)")
    plt.subplot(122), plt.imshow(img2_with_lines)
    plt.title("Epipolar Lines (Image 2)")

    plt.savefig(os.path.join('./output', key + '_epipolar-lines.png'), dpi = 300)

    return