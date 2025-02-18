import numpy as np
from triangulation import triangulation

#   Calculate the essential matrix
def get_essential_matrix(K1, K2, F):
    E = K1.T @ F @ K2
    u, s, v_T = np.linalg.svd(E)
    s = [1, 1, 0]
    E = u @ np.diag(s) @ v_T

    return E

#   Obtain 4 possible solutions of P2
def get_camera_matrix_choices(E, F):
    U, _, V_T = np.linalg.svd(E)
    W = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]])
    
    R_Arr = []
    t_Arr = []

    #   There will be 4 possible solutions in R_Arr and t_Arr
    R_Arr.append(U @ W @ V_T)
    R_Arr.append(U @ W @ V_T)
    R_Arr.append(U @ W.T @ V_T)
    R_Arr.append(U @ W.T @ V_T)

    t_Arr.append(U[:, 2:])
    t_Arr.append(- U[:, 2:])
    t_Arr.append(U[:, 2:])
    t_Arr.append(- U[:, 2:])

    for index in range(4):
        #   To check whether the determine of R is larger than 0
        if (np.linalg.det(R_Arr[index]) < 0):
            R_Arr[index] = - R_Arr[index]
            t_Arr[index] = - t_Arr[index]
    
    return R_Arr, t_Arr

#   Choose the solution of P2
def get_solution(pts1, pts2, K1, K2, R1, R2_Arr, t1, t2_Arr):
    #   Obtain the P1 matrix
    P1 = K1 @ np.hstack((R1, t1))

    #   Set the initial value
    P2_best = K2 @ np.hstack((R2_Arr[0], t2_Arr[0]))
    index_best = 0
    pointsNum_max = 0
    points3d_best = []

    for (R2, t2) in zip(R2_Arr, t2_Arr):
        #   Obtain the P2 matrix
        P2 = K2 @ np.hstack((R2, t2))
        #   Obtain the corresponding 3D points and the number of the 3D points in front of the cameras
        points3d, pointsNum = triangulation(pts1, pts2, P1, P2)

        #   Compare the number of the 3D points in front of the cameras
        #   To check whether P2 matrix is the best solution
        if pointsNum > pointsNum_max:
            pointsNum_max = pointsNum
            P2_best = P2
            points3d_best = points3d
    
    return points3d_best, pointsNum_max, P1, P2_best