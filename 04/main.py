import numpy as np
from load_data import *
from find_fundamental_matrix import *
from find_solution import *
from triangulation import texture_mapping, plot_points3d

def run(data_path):
    data = load_data(data_path)

    for key, (img1File, img2File, calib_path) in data.items():
        #   Read the image file
        img1 = cv2.imread(data_path + img1File)
        img2 = cv2.imread(data_path + img2File)

        #   Change the color channel from BGR to RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        #   Read the calibration file
        calibData = read_calib(data_path, calib_path)

        #   Find the good matches points in image1 and image2
        pts1, pts2, _ = find_correspondences(img1, img2)
        #   Obtain the fundamental matrix and the inlier points
        if data_path == 'my_data/':
            F, pts1, pts2 = RANSAC(pts1, pts2, iter_num = 1000000, error_threshold = 3.0, confidence = 0.5)
        else:
            F, pts1, pts2 = RANSAC(pts1, pts2)

        print(f'Number of correspondences: {len(pts1)}')
        print('Fundamental Matrix:')
        print(F)

        lines1, lines2 = compute_epipolar_line(F, pts1, pts2)
        img1_with_lines = get_lines(img1, lines1, pts1)
        img2_with_lines = get_lines(img2, lines2, pts2)

        draw_epipolar_lines(key, img1_with_lines, img2_with_lines)

        '''
        Set P1 = [R1 | t1] 
               = [1 0 0 0
                  0 1 0 0
                  0 0 1 0]
        '''
        R1 = np.identity(3)
        t1 = np.zeros((3, 1))
        E = get_essential_matrix(calibData.K1, calibData.K2, F)
        R2_Arr, t2_Arr = get_camera_matrix_choices(E, F)
        points3d, _, P1, P2 = get_solution(pts1, pts2, calibData.K1, calibData.K2, R1, R2_Arr, t1, t2_Arr)
        plot_points3d(key, points3d)

        if key == 'Mesona':
            index = 1.0
        elif key == 'Statue':
            index = 2.0
        else:
            index = 3.0
        texture_mapping(points3d, pts1, P1, data_path + img1File, key, im_index = index)

    return

if __name__ == "__main__":
    # data_path = 'data/'
    data_path = 'my_data/'
    run(data_path)