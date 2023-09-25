# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread('./image/block.jpg', cv.IMREAD_GRAYSCALE)

# Initiate ORB detector
orb = cv.ORB_create()

# Find the keypoints with ORB
kp = orb.detect(img,None)

# Compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# Draw only keypoints location, not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()