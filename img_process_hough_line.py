import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


""" Use Hough transform to detec lines in an image."""
# https://docs.opencv.org/4.8.0/d6/d10/tutorial_py_houghlines.html


# -------------------------------- Hough Transform --------------------------------------

img = cv.imread('./image/sudoku.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150,apertureSize = 3)
lines = cv.HoughLines(edges,1,np.pi/180,135)    # 4th arg: min votes to be considered as a line

for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv.imwrite('./image/sudoku-houghlines.jpg',img)



# --------------------------- Probabilistic Hough Transformation ------------------------
'''
It take a random subset of points which is sufficient for line detection, not all points
as in the hough transformation.
It is an optimization of the hough transformation. Have to decrease the threshold.

cv.HoughLinesP() returns two endpoints of lines (direct and simple)
'''

img = cv.imread('./image/sudoku.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150,apertureSize = 3)
lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
cv.imwrite('./image/sudoku-probhoughlines.jpg',img)
