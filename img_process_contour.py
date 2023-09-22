# Refs: https://docs.opencv.org/4.x/d3/d05/tutorial_py_table_of_contents_contours.html

import numpy as np
import cv2 as cv


# --------------------------------------- Draw Contours ----------------------------------

"""
Contour: 

A curve joining all continuous points along the boundary, having same color
or intensity. It is useful for shape analysis, object detection and recognition.

For better accuracy, use binary images.
Before finding contours, apply threshold or canny edge detection.

Objects to be found should be white and background should be black.
"""

img = cv.imread('./image/H.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

ret, thresh = cv.threshold(img, 127, 255, 0)

# contours: a list of all contours in the image
# 2nd arg: a contour retrieval mode
# 3rd arg: a contour approximation method
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

# Draw contours use cv.drawContours(img, contours, index of contours {-1 for all}, color, thickness, ...)
cv.drawContours(img, contours, -1, (0,255,0), 3)

# Draw an individual contour, e.x., 1st contour
cv.drawContours(img, contours, 0, (0,255,0), 3)

# Useful manner
cnt = contours[0]
cv.drawContours(img, [cnt], 0, (0,255,0), 3)

cv.imshow('Image with contours', img)
cv.waitKey(0)
cv.destroyAllWindows()


