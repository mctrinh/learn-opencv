import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# https://docs.opencv.org/4.x/dc/d7d/tutorial_py_brief.html

img = cv.imread('./image/block.jpg', cv.IMREAD_GRAYSCALE)

# Initiate FAST detector
star = cv.xfeatures2d.StarDetector_create()

# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

# Find the keypoints with STAR from CenSurE
kp = star.detect(img,None)

# Compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)

img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()

print(brief.descriptorSize())
print(des.shape)