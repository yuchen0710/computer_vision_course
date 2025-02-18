import numpy as np

def compute_homography(src_points, dst_points):
    """
    使用點對應關係手動計算單應矩陣 H
    src_points: 世界坐標點（棋盤格上的點）
    dst_points: 圖像平面點（角點坐標）
    """
    A = []

    for i in range(len(src_points)):
        X, Y = src_points[i][0], src_points[i][1]
        x, y = dst_points[i][0], dst_points[i][1]

        A.append([-X, -Y, -1, 0, 0, 0, x*X, x*Y, x])
        A.append([0, 0, 0, -X, -Y, -1, y*X, y*Y, y])

    A = np.array(A)

    # 使用 SVD 求解 Ah = 0
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    return H / H[2, 2]  # 將 H 進行歸一化處理，使得 H[2, 2] = 1

def cal_intrinsic_matrix(H):
    #   Get the system of Vb = 0
    #   Refer to the slide 77 to 80
    V = []
    for Hi in H:
        #   H is a 3 * 3 matrix
        #   H = (h1, h2, h3)
        h1 = Hi[:, 0]
        h2 = Hi[:, 1]
        
        #   Obtain the matrix V
        V1 = cal_V(h1, h2)
        V2 = cal_V(h1, h1) - cal_V(h2, h2)
        V.append(V1)
        V.append(V2)

    #   Obtain the vector b from the system Vb = 0 by using the SVD
    _, _, Vh = np.linalg.svd(V)
    Vh = Vh.T
    b = Vh[:, -1]

    #   Get the matrix B := K^-T * K^-1
    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]])
    
    #   To ensure the matrix B is symmetric and positive definite.
    if np.linalg.det(B) < 0:
        B = -B

    #   Obtain the intrinsic matrix, K.
    K = cal_K(B)

    return B, K

#   Since h1^T * K^-T * K^-1 * h2 = 0 and h1^T * K^-T * K^-1 * h1 = h2^T * K^-T * K^-1 * h2,
#   we can obtain the relation below.
def cal_V(h1, h2):
    V1 = h1[0] * h2[0]
    V2 = h1[0] * h2[1] + h1[1] * h2[0]
    V3 = h1[1] * h2[1]
    V4 = h1[2] * h2[0] + h1[0] * h2[2]
    V5 = h1[2] * h2[1] + h1[1] * h2[2]
    V6 = h1[2] * h2[2]

    return np.array([V1, V2, V3, V4, V5, V6])

#   Calculate the terms in the intrinsic matrix directly.
def cal_K(B):
    v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - (B[0, 1] ** 2))
    l = B[2, 2] - ((B[0, 2] ** 2) + v0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
    alpha = np.sqrt((l / B[0, 0]))
    beta = np.sqrt((l * B[0, 0] / (B[0, 0] * B[1, 1] - (B[0, 1] ** 2))))
    gamma = - B[0, 1] * (alpha ** 2) * beta / l
    u0 = (gamma * v0 / beta) - (B[0, 2] * (alpha ** 2) / l)

    K = np.array([[alpha,     0, u0],
                  [    0,  beta, v0],
                  [    0,     0,  1]])
    return K