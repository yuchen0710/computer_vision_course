import numpy as np
import random

def compute_homography(feature_img1, feature_img2):
    """
    使用點對應關係手動計算單應矩陣 H
    points1: 圖片1的點
    points2: 圖片2的點
    """
    A = []

    for i in range(len(feature_img2)):
        X, Y = feature_img2[i][0], feature_img2[i][1]
        x, y = feature_img1[i][0], feature_img1[i][1]

        A.append([-X, -Y, -1, 0, 0, 0, x * X, x * Y, x])
        A.append([0, 0, 0, -X, -Y, -1, y * X, y * Y, y])

    A = np.array(A)

    # 使用 SVD 求解 Ah = 0
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    return H / H[2, 2]  # 將 H 進行歸一化處理，使得 H[2, 2] = 1

def RANSAC(feature_img1, feature_img2, samples=10, iteration=10, threshold=5):
    best_score = 0
    best_homography = None
    for _ in range(iteration):
        indexes = random.sample(range(0, len(feature_img2)), samples)
        H = compute_homography(feature_img1[indexes], feature_img2[indexes])

        ones = np.ones((feature_img2.shape[0], 1))
        p1_homogeneous = np.hstack((feature_img2, ones)) # 轉為齊次座標
        result_points = [H @ p.T for p in p1_homogeneous]
        result_points = [p / p[2] for p in result_points] # 將Z軸變為1
        score = len([ a for a, b in zip(result_points, feature_img1) if np.linalg.norm(a[:2] - b) < threshold ]) # 計分
        # print(f'score: {score}')
        if score > best_score:
            best_score = score
            best_homography = H
    # print(f'best score: {best_score}')
    return best_homography