#!/usr/bin/env python3
"""
Image Stitching Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to stitch two images of overlap into one image.
To this end, you need to find feature points of interest in one image, and then find
the corresponding ones in another image. After this, you can simply stitch the two images
by aligning the matched feature points.
For simplicity, the input two images are only clipped along the horizontal direction, which
means you only need to find the corresponding features in the same rows to achieve image stiching.

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
"""
import cv2
import numpy as np
import random

# The threshold distance between two matches feature points
feature_dist_th = 0.5

# Projection threshold distance between two points for the RANSAC algorithm
ransac_proj_th = 5.0

def show_img(img, name="img", ms=1000):
    cv2.imshow(name, img)
    cv2.waitKey(ms)


def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def find_keypoints(img):
    """
    Finds the keypoints and descriptors for a grayscale image using SIFT
    """
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors


def find_best_matches(descriptors1, descriptors2, threshold):
    """
    Finds the best set of matches in two lists of descriptors of features
    """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    matches = np.asarray([m for m in matches\
            if m[0].distance < threshold * m[1].distance])
    return matches

############################################################################
##### opencv-python version 3.4.2.16 MUST BE USED IN ORDER TO USE SIFT #####
############################################################################
def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result image which is stitched by left_img and right_img
    """
    # ndarrays; shapes: (397, 529, 3) -> (397, 529)
    right_img_gray = to_grayscale(right_img)
    left_img_gray = to_grayscale(left_img)

    kp_right, des1 = find_keypoints(right_img_gray)
    kp_left, des2 = find_keypoints(left_img_gray)

    matches = find_best_matches(des1, des2, feature_dist_th)

    # Get coordinates of points in the original image plane
    src = np.float32([kp_right[m.queryIdx].pt for m in matches[:,0]])\
            .reshape(-1,1,2)

    # Get coordinates of points in the target image plane
    target = np.float32([kp_left[m.trainIdx].pt for m in matches[:,0]])\
            .reshape(-1,1,2)

    H, _ = cv2.findHomography(src, target, cv2.RANSAC, ransac_proj_th)

    # Transform right image to target image plane
    res = cv2.warpPerspective(right_img, H, (left_img.shape[1] +\
            right_img.shape[1], left_img.shape[0]))

    # Concatenate left image with tansformed right image
    res[0:left_img.shape[0], 0: left_img.shape[1]] = left_img
    
    return res 

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_image = solution(left_img, right_img)
    cv2.imwrite('results/task2_result.jpg',result_image)

