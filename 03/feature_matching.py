import cv2 as cv
import numpy as np
import copy

# Step 1: SIFT特徵檢測與描述
# 調整參數
def detect_and_describe(image, args):
    sift = cv.SIFT_create(nfeatures = args.nfeatures, contrastThreshold = args.contrastThreshold, edgeThreshold = args.edgeThreshold, sigma = args.sigma)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    img_keypoints = cv.drawKeypoints(image, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return keypoints, descriptors, img_keypoints

# Step 2: 特徵匹配
# 調整參數ratio_test
def match_features(descriptors1, descriptors2, img1, img2, keypoints1, keypoints2, args):
    matches = []
    for index_1 in range(len(descriptors1)):
        distance_min = [-1, float('inf')]
        distance_sec = [-1, float('inf')]
        for index_2 in range(len(descriptors2)):
            distance = np.linalg.norm(descriptors1[index_1] - descriptors2[index_2])
            if distance < distance_min[1]:
                distance_sec = copy.copy(distance_min)
                distance_min = [index_2, distance]
            elif (distance < distance_sec[1]) and (distance != distance_min[1]):
                distance_sec = [index_2, copy.copy(distance)]
        
        if distance_min[1] < distance_sec[1] * args.ratio_dis:
            # matches.append((index_1, distance_min[0]))
            match = cv.DMatch()
            match.queryIdx = index_1
            match.trainIdx = distance_min[0]
            match.distance = distance_min[1]
            matches.append(match)

    img_matching = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    feature_img1 = []
    feature_img2 = []

    for term in matches:
        feature_img1.append(keypoints1[term.queryIdx].pt)
        feature_img2.append(keypoints2[term.trainIdx].pt)
    feature_img1 = np.array(feature_img1)
    feature_img2 = np.array(feature_img2)

    return feature_img1, feature_img2, img_matching