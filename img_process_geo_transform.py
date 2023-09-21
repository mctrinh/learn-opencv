import cv2 as cv
import numpy as np


# ------------------------------- Scaling -----------------------------------
""" Two transformation functions can perform all kinds of transformations:
        cv.warpAffine: take a 2x3 transformation matrix as input
        cv.warpPerspective(): take a 3x3 transformation matrix as input
"""

img = cv.imread('./image/messi.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
cv.imshow('img', img)
cv.waitKey(0)

res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
cv.imshow('res', res)
cv.waitKey(0)
# Or
height, width = img.shape[:2]
res1 = cv.resize(img, (2*width, 2*height), interpolation=cv.INTER_CUBIC)
cv.imshow('res1', res1)
cv.waitKey(0)


# ------------------------------- Translation -----------------------------------
""" Let the shift in (x,y) direction is (tx,ty)
    The transformation matrix:
               M = [[1 0 tx],
                   [0 1 ty]]
"""
img = cv.imread('./image/messi.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
rows, cols = img.shape

M = np.float32([[1,0,100], [0,1,50]])
dst = cv.warpAffine(img, M, (cols,rows))    # 3rd arg is the size of the output image (width = no. of colums, height = no. of rows)
cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()


# ------------------------------- Rotation -----------------------------------
""" Rotation of an image for an angle theta, the transformation matrix:
            M = [[cos(theta)  -sin(theta)],
                 [sin(theta)   cos(theta)]]

Can rotate at any location, the modified transformation matrix is:
            M = [[alpha  beta   (1-alpha)*center_x-beta*center_y]
                 [-beta  alpha  beta*center_x+(1-alpha)*center_y]]
    where:
            alpha = scale*cos(theta)
            beta  = scale*sin(theta)
OpenCV:
            M = cv.getRotationMatrix2D((center_x,center_y), theta, scale)
"""
# cols-1 and rows-1 are the coordinate limits
M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 90, 1)
dst = cv.warpAffine(img, M, (cols,rows))
cv.imshow('dst', dst)
cv.waitKey(0)


# ------------------------------- Affine Transformation -----------------------------------
""" In affine transformation, all parallel lines in the original image still be parallel
    in the output image. To find the transformation matrix, we need three points from 
    the input image and their corresponding locations in the output image.

    cv.getAffineTransform() will create a 2x3 transformation matrix which is passed
        to cv.warpAffine()
"""

img = cv.imread('./image/card.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
rows, cols, ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv.getAffineTransform(pts1,pts2)

dst = cv.warpAffine(img,M,(cols,rows))

import matplotlib.pyplot as plt
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()


# ------------------------------- Perspective Transformation -----------------------------------
""" In affine transformation, straight lines remain straight after transformation. To find the
    transformation matrix (3x3), need 4 points on the input image and corresponding points on
    the output image. Among these 4 points, 3 of them should not be colinear.

    cv.getPerspectiveTransform() will create a 3x3 transformation matrix, which is passed
        to cv.warpPerspective()
"""

img = cv.imread('./image/sudoku.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
rows, cols, ch = img.shape

pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv.getPerspectiveTransform(pts1,pts2)

dst = cv.warpPerspective(img,M,(300,300))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
