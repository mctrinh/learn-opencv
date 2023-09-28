# Ref: https://docs.opencv.org/4.8.0/dc/dbb/tutorial_py_calibration.html

import numpy as np
import cv2 as cv
import glob

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane
images = glob.glob('./image/left*[0-1][0-9].jpg*')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If the desired number of corners are found in the img then ret=true
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        # Refining pixel coordinates for given 2d points
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
    cv.imshow('img', img)
    cv.waitKey(1000)
cv.destroyAllWindows()

h,w = img.shape[:2]

"""
Performing camera calibration by passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the detected corners (imgpoints). """

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)


# -------------------------------- Undistortion --------------------------------

img = cv.imread('./image/left12.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# cv.undistort() is the easiest way:
dst = cv.undistort(img, mtx, dist, None, newcameramtx)      # undistort
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]     # crop the image
cv.imwrite('./output/calibresult_1.jpg', dst)

# Find a mapping function from the distorted image to the undistorted image, then use remap function.
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)    # undistort
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
x, y, w, h = roi            # crop the image
dst = dst[y:y+h, x:x+w]
cv.imwrite('./output/calibresult_2.jpg', dst)


# -------------------------------- Re-projection Error --------------------------------

""" A good estimation of how exact the found parameters are.
    Given the intrinsic, distortion, rotation and translation matrices,
    we must first transform the object point to image point using cv.projectPoints()
    Then, we can calculate the absolute norm between what we got with our transformation
    and the corner finding algorithm. To find the average error, we calculate
    the arithmetical mean of the errors calculated for all the calibration images.
"""

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
