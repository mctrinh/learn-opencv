import cv2 as cv
import numpy as np


# --------------------------- Pyramids = A stack of different resolutions ------------------
"""
Create a set of the same image with different resolutions and search for object in all of them.
Stack newly created imnages in the vertical direction bot -> top : high -> low resolutions.

Two kinds of image pyramids: 1) Gaussian Pyramid, 2) Laplacian Pyramids

"""

img = cv.imread('./image/messi.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"

lower_reso = cv.pyrDown(img)                    # If dont specify, cv.pyrDown will use the default dstsize = ((src.cols+1)/2, (src.rows+1)/2)
cv.imshow('Lower Resolution', lower_reso)       # lower_reso looses information
cv.waitKey(0)

higher_reso = cv.pyrUp(lower_reso)
cv.imshow('Higher Resolution', higher_reso)     # higher_reso is different from img
cv.waitKey(0)
cv.destroyAllWindows()


# ------------------------------ Image Blending using Pyramids -----------------------------

A = cv.imread('./image/apple.png')
B = cv.imread('./image/orange.png')
assert A is not None, "file could not be read, check with os.path.exists()"
assert B is not None, "file could not be read, check with os.path.exists()"
cv.imshow('Apple', A)
cv.imshow('Orange', B)
cv.waitKey(0)

# Generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv.pyrDown(gpA[i])              # choose the last downsampled image
    gpA.append(G)

# Generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv.pyrDown(gpB[i])              # choose the last downsampled image
    gpB.append(G)

"""
Now the script works for images of even shape, but not an odd shape, as cv2.pyrDown
computes the default size. We have to give to cv2.pyrUp the proper dstsize parameter
according to the image that you use to do the cv2.substract (or cv2.add)
"""

# Generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5,0,-1):
    size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
    GE = cv.pyrUp(gpA[i], dstsize=size)
    L = cv.subtract(gpA[i-1],GE)
    lpA.append(L)

# Generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5,0,-1):
    size = (gpB[i-1].shape[1], gpB[i-1].shape[0])
    GE = cv.pyrUp(gpB[i], dstsize=size)
    L = cv.subtract(gpB[i-1],GE)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
    LS.append(ls)

# Now reconstruct
ls_ = LS[0]
for i in range(1,6):
    size = (LS[i].shape[1], LS[i].shape[0])
    ls_ = cv.pyrUp(ls_, dstsize=size)
    ls_ = cv.add(ls_, LS[i])

# Image with direct connecting each half
real = np.hstack((A[:,:cols//2],B[:,cols//2:]))
cv.imwrite('./output/Pyramid_blending.jpg',ls_)
cv.imwrite('./output/Direct_blending.jpg',real)
cv.imshow('Pyramid blending', ls_)
cv.imshow('Direct blending', real)
cv.waitKey(0)
cv.destroyAllWindows()

