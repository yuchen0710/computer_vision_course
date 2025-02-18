import cv2
import numpy as np
import os

window_size = 30

def shift(matrix, offset):
    shifted_matrix = shift_X(matrix, offset[0])
    shifted_matrix = shift_Y(shifted_matrix, offset[1])
    return shifted_matrix

def shift_Y(matrix, amount):
    shifted_matrix = np.zeros_like(matrix)
    if amount > 0:
        shifted_matrix[amount:, :] = matrix[:-amount, :]
    elif amount < 0:
        shifted_matrix[:amount, :] = matrix[-amount:, :]
    else:
        return matrix
    return shifted_matrix

def shift_X(matrix, amount):
    shifted_matrix = np.zeros_like(matrix)
    if amount > 0:
        shifted_matrix[:, amount:] = matrix[:, :-amount]
    elif amount < 0:
        shifted_matrix[:, :amount] = matrix[:, -amount:]
    else:
        return matrix
    return shifted_matrix

def cut_image(image):
    h = int(image.shape[0] / 3)
    return image[:h], image[h:h*2], image[h*2:h*3]

def SSD(source, target, init_offset=(0, 0), max_window_size=50):
    global window_size

    _window_size = min(window_size, max_window_size)
    center_x = int(source.shape[1] / 2)
    center_y = int(source.shape[0] / 2)

    RADIUS = 80
    upper_left_X = center_x - RADIUS
    upper_left_Y = center_y - RADIUS
    patch = source[center_y-RADIUS:center_y+RADIUS, center_x-RADIUS:center_x+RADIUS]

    min_SSD = 1e+20
    min_coordinate = (0, 0)
    for y in range(_window_size):
        print(f'Evaluating... [{y + 1} / {_window_size}]\r', end='')
        for x in range(_window_size):
            y_start = upper_left_Y - int(_window_size / 2) + y + init_offset[1]
            x_start = upper_left_X - int(_window_size / 2) + x + init_offset[0]

            sum = 0.0
            for b in range(patch.shape[0]):
                for a in range(patch.shape[1]):
                    diff = int(patch[b][a]) - int(target[y_start + b][x_start + a])
                    sum += diff ** 2

            if sum < min_SSD:
                min_SSD = sum
                min_coordinate = (x, y)
    offset = (min_coordinate[0] - int(_window_size / 2), min_coordinate[1] - int(_window_size / 2))
    print(f'SSD: returned offset {offset}')
    return offset

def run():
    SCALE = 8
    try:
        os.mkdir('colorizing_output')
        print(f"Directory created successfully.")
    except FileExistsError:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")

    for filename in os.listdir(os.getcwd() + '/data/task3_colorizing'):
        print(f'\nProcessing {filename}...')
        img = cv2.imread('data/task3_colorizing/' + filename, cv2.IMREAD_GRAYSCALE)

        if '.tif' in filename:
            new_size = (int(img.shape[1] / SCALE), int(img.shape[0] / SCALE))
            img = cv2.resize(img, new_size)

        img_B, img_G, img_R = cut_image(img)

        offset_R = SSD(img_R, img_G)
        offset_B = SSD(img_B, img_G)
        shifted_R = shift(img_R, offset_R)
        shifted_B = shift(img_B, offset_B)

        img_BGR = cv2.merge([shifted_B, img_G, shifted_R])

        if '.tif' in filename:
            img = cv2.imread('in/' + filename, cv2.IMREAD_GRAYSCALE)
            img_B, img_G, img_R = cut_image(img)

            offset_R = (offset_R[0] * SCALE, offset_R[1] * SCALE)
            offset_B = (offset_B[0] * SCALE, offset_B[1] * SCALE)
            offset_R_precise = SSD(img_R, img_G, offset_R, SCALE)
            offset_B_precise = SSD(img_B, img_G, offset_B, SCALE)
            shifted_R = shift(img_R, (offset_R[0]+offset_R_precise[0], offset_R[1]+offset_R_precise[1]))
            shifted_B = shift(img_B, (offset_B[0]+offset_B_precise[0], offset_B[1]+offset_B_precise[1]))

            img_BGR = cv2.merge([shifted_B, img_G, shifted_R])

        if '.jpg' in filename:
            img_path = 'colorizing_output/' + filename.replace('.jpg', '_out.jpg')
        else:
            img_path = 'colorizing_output/' + filename.replace('.tif', '_out.jpg')
        if not os.path.exists(img_path):
            cv2.imwrite(img_path, img_BGR, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print('Image saved successfully!')
        else:
            print('Image already exists!')
    exit()

if __name__ == '__main__':
    run()
    