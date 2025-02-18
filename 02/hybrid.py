import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

''' Calculate the distance from the point to the center '''
def cal_distance(center, point):
    return np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)

''' Define the ideal low pass filter function '''
def ideal_low_pass_mask(row, column, cutoff):
    mask = np.zeros((row, column))

    #   Get the center
    center_x, center_y = row / 2, column / 2

    for x in range(row):
        for y in range(column):
            if (cal_distance([center_x, center_y], [x, y]) <= cutoff):
                mask[x, y] = 1

    return mask

''' Define the Gaussian low pass filter function '''
def gaussian_low_pass_mask(row, column, cutoff):
    mask = np.zeros((row, column))

    #   Get the center
    center_x, center_y = row / 2, column / 2

    for x in range(row):
        for y in range(column):
            mask[x, y] = np.exp(((- cal_distance([center_x, center_y], [x, y]) ** 2)/(2 * (cutoff ** 2))))

    return mask

''' Define the ideal high pass filter function '''
def ideal_high_pass_mask(row, column, cutoff):
    mask = np.zeros((row, column))

    #   Get the center
    center_x, center_y = row / 2, column / 2

    for x in range(row):
        for y in range(column):
            if (cal_distance([x, y], [center_x, center_y]) >= cutoff):
                mask[x, y] = 1

    return mask

''' Define the Gaussian high pass filter function '''
def gaussian_high_pass_mask(row, column, cutoff):
    mask = np.zeros((row, column))

    #   Get the center
    center_x, center_y = row / 2, column / 2

    for x in range(row):
        for y in range(column):
                mask[x, y] = 1 - np.exp(((- cal_distance([center_x, center_y], [x, y]) ** 2)/(2 * (cutoff ** 2))))

    return mask

def image_filter(img, mask):
    #   Split the RGB channel to the variables
    img_b, img_g, img_r = cv2.split(img)
    #   2D Fourier transform
    b_fft, g_fft, r_fft = np.fft.fft2(img_b), np.fft.fft2(img_g), np.fft.fft2(img_r)
    #   Shift the 0 frequency to the center
    b_shift, g_shift, r_shift = np.fft.fftshift(b_fft), np.fft.fftshift(g_fft), np.fft.fftshift(r_fft)

    b_filter = b_shift * mask
    g_filter = g_shift * mask
    r_filter = r_shift * mask

    return b_filter, g_filter, r_filter

def hybrid_image(img1, img2, mask_lpf, mask_hpf):
    #   Low pass filter
    b_lpf, g_lpf, r_lpf = image_filter(img1, mask_lpf)

    #   High pass filter
    b_hpf, g_hpf, r_hpf = image_filter(img2, mask_hpf)

    #   Combine the result
    b_hybrid = b_lpf + b_hpf
    g_hybrid = g_lpf + g_hpf
    r_hybrid = r_lpf + r_hpf

    #   Inverse of fftshift
    b_shift, g_shift, r_shift = np.fft.ifftshift(b_hybrid), np.fft.ifftshift(g_hybrid), np.fft.ifftshift(r_hybrid)
    #   Inverse 2D Fourier transform
    b_ifft, g_ifft, r_ifft = np.fft.ifft2(b_shift), np.fft.ifft2(g_shift), np.fft.ifft2(r_shift)

    #   Calculate the magnitude
    b_abs = np.abs(b_ifft)
    g_abs = np.abs(g_ifft)
    r_abs = np.abs(r_ifft)

    #   Normalize the value to the range 0 ~ 255
    img_b = cv2.normalize(b_abs, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
    img_g = cv2.normalize(g_abs, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
    img_r = cv2.normalize(r_abs, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)

    #   Merge the channels
    img_hybrid = cv2.merge((img_r, img_g, img_b))

    #   Generate low-pass filter image
    b_lpf, g_lpf, r_lpf = np.fft.ifftshift(b_lpf), np.fft.ifftshift(g_lpf), np.fft.ifftshift(r_lpf)
    b_lpf, g_lpf, r_lpf = np.fft.ifft2(b_lpf), np.fft.ifft2(g_lpf), np.fft.ifft2(r_lpf)
    b_lpf, g_lpf, r_lpf = np.abs(b_lpf), np.abs(g_lpf), np.abs(r_lpf)

    b_lpf = cv2.normalize(b_lpf, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
    g_lpf = cv2.normalize(g_lpf, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
    r_lpf = cv2.normalize(r_lpf, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
    img_lpf = cv2.merge((b_lpf, g_lpf, r_lpf))

    #   Generate high-pass filter image
    b_hpf, g_hpf, r_hpf = np.fft.ifftshift(b_hpf), np.fft.ifftshift(g_hpf), np.fft.ifftshift(r_hpf)
    b_hpf, g_hpf, r_hpf = np.fft.ifft2(b_hpf), np.fft.ifft2(g_hpf), np.fft.ifft2(r_hpf)
    b_hpf, g_hpf, r_hpf = np.abs(b_hpf), np.abs(g_hpf), np.abs(r_hpf)

    b_hpf = cv2.normalize(b_hpf, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
    g_hpf = cv2.normalize(g_hpf, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
    r_hpf = cv2.normalize(r_hpf, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
    img_hpf = cv2.merge((b_hpf, g_hpf, r_hpf))

    return img_hybrid, img_lpf, img_hpf

if __name__ == '__main__':
    ''' Fetch and pair the images '''
    print('Find the data...')
    #   Set the road to the data folder
    folderPath = 'data/task1and2_hybrid_pyramid/'
    # folderPath = 'my_data/'
    images = os.listdir(folderPath)

    print('Pair the images...')
    imagePair = {}
    for fileName in images:
        if fileName[0] in imagePair:
            imagePair[fileName[0]].append(fileName)
        else:
            imagePair[fileName[0]] = [fileName]

    #   Ensure every pair has only two images
    imagePair = {key: pair for key, pair in imagePair.items() if len(pair) == 2}

    #   Set the filter type
    gen_lpf_mask = gaussian_low_pass_mask
    gen_hpf_mask = gaussian_high_pass_mask
    filterType = '_gaussian'
    # gen_lpf_mask = ideal_low_pass_mask
    # gen_hpf_mask = ideal_high_pass_mask
    # filterType = '_ideal'

    print('Start to generate the hybrid images...')
    for key, (file1, file2) in imagePair.items():
        #   Read the data
        img1 = cv2.imread(folderPath + file1)
        img2 = cv2.imread(folderPath + file2)

        #   Change the varible type from 'uint8' to int
        img1 = img1.astype(int)
        img2 = img2.astype(int)

        #   Make sure the images have the same size
        if img1.shape[0] <= img2.shape[0]:
            row = img1.shape[0]
        else:
            row = img2.shape[0]
        
        if img1.shape[1] <= img2.shape[1]:
            column = img1.shape[1]
        else:
            column  = img2.shape[1]
        
        img1 = img1[0 : row, 0 : column, :]
        img2 = img2[0 : row, 0 : column, :]
            
        #   Generate the filter mask
        mask_lpf = gen_lpf_mask(row, column, 20)
        mask_hpf = gen_hpf_mask(row, column, 25)

        #   Get the hybrid image, low-pass filter image and high-pass filter image
        img_hybrid, img_lpf, img_hpf = hybrid_image(img1, img2, mask_lpf, mask_hpf)

        #   Save the images
        if folderPath == 'my_data/':
            plt.imsave('./output/' + key + filterType + '_hybrid_mydata' + file1[-4:], img_hybrid.astype('uint8'))
            plt.imsave('./output/low_pass_filter/' + key + filterType + 'Lpf_mydata' + file1[-4:], img_lpf.astype('uint8'))
            plt.imsave('./output/high_pass_filter/' + key + filterType + 'Hpf_mydata' + file1[-4:], img_hpf.astype('uint8'))
        else:
            plt.imsave('./output/' + key + filterType + '_hybrid' + file1[-4:], img_hybrid.astype('uint8'))
            plt.imsave('./output/low_pass_filter/' + key + filterType + 'Lpf' + file1[-4:], img_lpf.astype('uint8'))
            plt.imsave('./output/high_pass_filter/' + key + filterType + 'Hpf' + file1[-4:], img_hpf.astype('uint8'))
    print("Finish!")