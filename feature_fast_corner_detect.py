import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# https://docs.opencv.org/4.x/df/d0c/tutorial_py_fast.html
"""
Some algorithms are not fast enough for real-time apps.
FAST is features from accelerated segment test.
"""

img = cv.imread('./image/shapes.jpg', cv.IMREAD_GRAYSCALE)

# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()

# Find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

# Print all default params
print("Threshold: {}".format(fast.getThreshold()))
print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
print("neighborhood: {}".format(fast.getType()))
print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))

cv.imwrite('./image/shapes-fast.jpg', img2)

# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)
print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
cv.imwrite('./image/shapes-fast-false.jpg', img3)