import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 創建資料夾，如果不存在則創建
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 創建高斯金字塔
def create_gaussian_pyramid(image, levels=5):
    gaussian_pyramid = [image]
    for i in range(1, levels):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    return gaussian_pyramid

# 創建拉普拉斯金字塔
def create_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1, 0, -1):
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i])
        gaussian_expanded = cv2.resize(gaussian_expanded, (gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0]))
        laplacian = cv2.subtract(gaussian_pyramid[i-1], gaussian_expanded)
        laplacian_pyramid.append(laplacian)
    return laplacian_pyramid

# 計算圖像的頻譜
def compute_spectrum(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return magnitude_spectrum

# 保存圖像金字塔的每一層
def save_pyramid(pyramid, output_dir, title):
    ensure_dir(output_dir)
    for i, layer in enumerate(pyramid):
        filename = os.path.join(output_dir, f'{title}_level_{i}.png')
        cv2.imwrite(filename, layer)

# 保存頻譜圖
def save_spectrum_pyramid(pyramid, output_dir, title):
    ensure_dir(output_dir)
    for i, layer in enumerate(pyramid):
        plt.imshow(layer, cmap='gray')
        filename = os.path.join(output_dir, f'{title}_spectrum_level_{i}.png')
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

def main():
    image_paths = [
        'C:/Users/User/Desktop/computer_vision/CV2024_HW2/data/task1and2_hybrid_pyramid/0_Afghan_girl_after.jpg',
        'C:/Users/User/Desktop/computer_vision/CV2024_HW2/data/task1and2_hybrid_pyramid/0_Afghan_girl_before.jpg',
        'C:/Users/User/Desktop/computer_vision/CV2024_HW2/data/task1and2_hybrid_pyramid/1_bicycle.bmp',
        'C:/Users/User/Desktop/computer_vision/CV2024_HW2/data/task1and2_hybrid_pyramid/1_motorcycle.bmp',
        'C:/Users/User/Desktop/computer_vision/CV2024_HW2/data/task1and2_hybrid_pyramid/2_bird.bmp',
        'C:/Users/User/Desktop/computer_vision/CV2024_HW2/data/task1and2_hybrid_pyramid/2_plane.bmp',
        'C:/Users/User/Desktop/computer_vision/CV2024_HW2/data/task1and2_hybrid_pyramid/3_cat.bmp',
        'C:/Users/User/Desktop/computer_vision/CV2024_HW2/data/task1and2_hybrid_pyramid/3_dog.bmp',
        'C:/Users/User/Desktop/computer_vision/CV2024_HW2/data/task1and2_hybrid_pyramid/4_einstein.bmp',
        'C:/Users/User/Desktop/computer_vision/CV2024_HW2/data/task1and2_hybrid_pyramid/4_marilyn.bmp',
        'C:/Users/User/Desktop/computer_vision/CV2024_HW2/data/task1and2_hybrid_pyramid/5_fish.bmp',
        'C:/Users/User/Desktop/computer_vision/CV2024_HW2/data/task1and2_hybrid_pyramid/5_submarine.bmp',
        'C:/Users/User/Desktop/computer_vision/CV2024_HW2/data/task1and2_hybrid_pyramid/6_makeup_after.jpg',
        'C:/Users/User/Desktop/computer_vision/CV2024_HW2/data/task1and2_hybrid_pyramid/6_makeup_before.jpg'
    ]


    output_base_dir = 'C:/Users/User/Desktop/computer_vision/CV2024_HW2/data/pyramid_output'

    for image_path in image_paths:
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        image_name = os.path.basename(image_path).split('.')[0]

        # 創建高斯金字塔
        gaussian_pyramid = create_gaussian_pyramid(image, levels=5)
        gaussian_output_dir = os.path.join(output_base_dir, f'{image_name}_gaussian_pyramid')
        save_pyramid(gaussian_pyramid, gaussian_output_dir, 'Gaussian')

        # 創建拉普拉斯金字塔
        laplacian_pyramid = create_laplacian_pyramid(gaussian_pyramid)
        laplacian_output_dir = os.path.join(output_base_dir, f'{image_name}_laplacian_pyramid')
        save_pyramid(laplacian_pyramid, laplacian_output_dir, 'Laplacian')

        # 計算並保存高斯金字塔的頻譜
        gaussian_spectrum = [compute_spectrum(layer) for layer in gaussian_pyramid]
        gaussian_spectrum_output_dir = os.path.join(output_base_dir, f'{image_name}_gaussian_spectrum')
        save_spectrum_pyramid(gaussian_spectrum, gaussian_spectrum_output_dir, 'Gaussian')

        # 計算並保存拉普拉斯金字塔的頻譜
        laplacian_spectrum = [compute_spectrum(layer) for layer in laplacian_pyramid]
        laplacian_spectrum_output_dir = os.path.join(output_base_dir, f'{image_name}_laplacian_spectrum')
        save_spectrum_pyramid(laplacian_spectrum, laplacian_spectrum_output_dir, 'Laplacian')

        print(f"Processed and saved Gaussian and Laplacian pyramids and spectra for {image_name}")

if __name__ == "__main__":
    main()
