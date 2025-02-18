import matplotlib.pyplot as plt

def draw_keypoints(outputpath, key, img1, img2):
    plt.clf()
    plt.figure(figsize = (8, 8))
    plt.imshow(img1)
    plt.title(f"{key}1 - Keypoints")
    plt.axis("on")
    plt.savefig(f"{outputpath}{key}1_keypoints.jpg", bbox_inches='tight', pad_inches=0.1)

    plt.figure(figsize = (8, 8))
    plt.imshow(img2)
    plt.title(f"{key}2 - Keypoints")
    plt.axis("on")
    plt.savefig(f"{outputpath}{key}2_keypoints.jpg", bbox_inches='tight', pad_inches=0.1)

    return

def draw_feature_matching(outputpath, key, image):
    plt.clf()
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title(f"{key} - Feature Matches")
    plt.axis("on")
    plt.savefig(f"{outputpath}{key}_feature_matching.jpg", bbox_inches='tight', pad_inches=0.1)

    return

def draw_result(outputpath, key, image):
    plt.clf()
    plt.imshow(image)
    # plt.title(f"{key} - Final Result")
    plt.axis("off")
    plt.savefig(f"{outputpath}{key}_final_result.jpg", bbox_inches='tight', pad_inches=0)
    return