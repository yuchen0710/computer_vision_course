import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
import os

def triangulation(pts1, pts2, P1, P2):
    length = len(pts1)
    points3d = np.ones((length, 4))
    C = P2[:, :3].T @ P2[:, 3:]
    num_inFront = 0
    
    for index in range(length):
        A = np.array([
            pts1[index, 0] * P1[2, :].T - P1[0, :].T,
            pts1[index, 1] * P1[2, :].T - P1[1, :].T,
            pts2[index, 0] * P2[2, :].T - P2[0, :].T,
            pts2[index, 1] * P2[2, :].T - P2[1, :].T
        ])
        _, _, V_T = np.linalg.svd(A)
        points = V_T[-1] / V_T[-1, 3]
        points3d[index, :] = points

        if np.dot(points[:3] - C.reshape(-1), P2[2, :3]) > 0:
            num_inFront += 1
    
    return points3d, num_inFront

def plot_points3d(key, points3d, output_path = 'output'):
    plt.clf()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], c = 'blue', marker='o')
    ax.set_title('3D points of ' + key)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')

    fig.savefig(os.path.join(output_path, key + '_3Dpoints.png'), dpi = 300)

    return

def texture_mapping(points3d, pts1, P1, imgFile, key, im_index = 1, output_path = 'output'):
    """
    Args:
        P2 (numpy.ndarray): 第二台相機的投影矩陣 (3x4)
        points1 (numpy.ndarray): 第一台相機中的像素坐標點 (Nx2)
        points2 (numpy.ndarray): 第二台相機中的像素坐標點 (Nx2)
        im_index (int): 圖片編號
    """
    eng = matlab.engine.start_matlab()
    eng.obj_main(
        matlab.double(points3d[:, :3].tolist()), matlab.double(pts1.tolist()), matlab.double(P1.tolist()),
        './{}'.format(imgFile), im_index, output_path, nargout = 0
    )
    os.rename(f'model{int(im_index)}.mtl', os.path.join(output_path, f'model_{key}.mtl'))
    os.rename(f'model{int(im_index)}.obj', os.path.join(output_path, f'model_{key}.obj'))