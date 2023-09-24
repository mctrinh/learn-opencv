import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


""" Marker-based image segmentation using watershed algorithm."""
# https://docs.opencv.org/4.8.0/d3/db4/tutorial_py_watershed.html

img = cv.imread('./image/coins.jpeg')
assert img is not None, "file could not be read, check with os.path.exists()"
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

cv.imshow("Thresh", thresh)
cv.waitKey(0)

# Noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
cv.imshow("Opening", opening)
cv.waitKey(0)

# Sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)
cv.imshow("sure_bg", sure_bg)
cv.waitKey(0)

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
cv.imshow("sure_fg", sure_fg)
cv.waitKey(0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
cv.imshow("unknown", unknown)
cv.waitKey(0)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

# Final step, apply watershed: the marker image will be modified, boundary region is marked with -1
markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]
cv.imshow("segmented image", img)
cv.waitKey(0)

