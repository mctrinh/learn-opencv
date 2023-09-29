# https://docs.opencv.org/4.8.0/d7/d53/tutorial_py_pose.html

import numpy as np
import cv2 as cv
import glob


# Load the camera matrix and distortion coefficients from calibration result
with np.load('./output/cam_calibration.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]



# Draw a 3D axis from corners in the chessboard and axis points
def draw_axis(img, corners, imgpts):
    a,b = corners[0].ravel()
    c,d = imgpts[0].ravel()
    e,f = imgpts[1].ravel()
    g,h = imgpts[2].ravel()
    img = cv.line(img, (int(a),int(b)), (int(c),int(d)), (255,0,0), 5)  # x blue
    img = cv.line(img, (int(a),int(b)), (int(e),int(f)), (0,255,0), 5)  # y green
    img = cv.line(img, (int(a),int(b)), (int(g),int(h)), (0,0,255), 5)  # z red
    return img

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)       # axis of length 3, (x,y,-z)

# Load each image, search for 7x6 grid, if found, refine it with subcorner pixels
for fname in glob.glob('./image/left*.jpg'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7,6),None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw_axis(img,corners2,imgpts)
        cv.imshow('img',img)
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite('./output/'+fname[:6]+'.png', img)
cv.destroyAllWindows()



# Render a cube
def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])    # 8 corners of a cube in 3D space

# Load each image, search for 7x6 grid, if found, refine it with subcorner pixels
for fname in glob.glob('./image/left*.jpg'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7,6),None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw_cube(img,corners2,imgpts)
        cv.imshow('img',img)
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite('./output/'+fname[:6]+'.png', img)
cv.destroyAllWindows()


