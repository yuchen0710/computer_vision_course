import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import argparse
from feature_matching import detect_and_describe, match_features
from ransac import RANSAC
from warp import warp
from draw_image import draw_keypoints, draw_feature_matching, draw_result

def run(args):
    images = os.listdir(args.input_path)
    #   Check whether the folder exists
    if not os.path.isdir(args.input_path):
        raise FileNotFoundError("Folder not found.")
    
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path, mode=0o777, exist_ok=True) 

    print("Pair the image...")

    image_pair = {}
    
    for file in images:
        filename, filetype = file.split('.')
        if filename[:-1] in image_pair:
            continue
        else:
            image_pair[filename[:-1]] = [filename[:-1] + '1.' + filetype, filename[:-1] + '2.' + filetype]
    print(image_pair)
    
    for key, (file1, file2) in image_pair.items():
        img1 = cv2.imread(args.input_path + file1)
        img2 = cv2.imread(args.input_path + file2)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        print("Feature matching...")
        # 特徵點檢測和描述
        keypoints1, descriptors1, img1_keypoints = detect_and_describe(img1, args)
        keypoints2, descriptors2, img2_keypoints = detect_and_describe(img2, args)

        draw_keypoints(args, key, img1_keypoints, img2_keypoints)

        # 特徵匹配
        feature_img1, feature_img2, img_matching = match_features(descriptors1, descriptors2, img1, img2, keypoints1, keypoints2, args)
        draw_feature_matching(args, key, img_matching)

        print("Find homorgraphy matrix H...")
        H = RANSAC(feature_img1, feature_img2, 15, 15, 0.3)

        print("Warp the images...")
        image = warp(img1, img2, H, args)
        draw_result(args, key, image)
    print("Finish!")

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #   Set the path of the input and output image folder
    parser.add_argument('--input_path', type=str, default='./my_data/', choices=['./data/', './my_data/'], help='Input Data Path')
    parser.add_argument('--output_path', type=str, default='./output/', help='Output Data Path')
    #   Set the features matching parameters
    parser.add_argument('--nfeatures', type=int, default=1000, help='number of the features')
    parser.add_argument('--contrastThreshold', type=float, default=0.04, help='Contrast Threshold')
    parser.add_argument('--edgeThreshold', type=int, default=10, help='Edge Threshold')
    parser.add_argument('--sigma', type=float, default=1.6, help='sigma')
    #   Set the distance ratio
    parser.add_argument('--ratio_dis', type=float, default=0.5, help='ratio distance')
    #   Set the blending mode and the relative parameters
    parser.add_argument('--mode', type=str, default='linear', choices=['fix', 'sigmoid', 'linear'], help='blending mode')
    parser.add_argument('--blend_threshold', type=int, default=50, help='blending mode')
    args = parser.parse_args()

    run(args)