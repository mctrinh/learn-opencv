import cv2 as cv
import numpy as np


# ------------------------------- Image Addition -----------------------------------
# cv.add() to add two images (saturated operation)
# or simply by the numpy operation res = img1 + img2 (modulo operation)
# both images should be of same depth, or 2nd image can be a scalar value.

x = np.uint8([250])
y = np.uint8([10])
print(cv.add(x,y))  # 250+10=260 -> 255

print(x+y)  # 250+10=260 % 256 = 4 -> cd.add() provide a better result.


# ------------------------------- Image Blending -----------------------------------
# Image addition with different weights, give a feeling of blending or transparency.
# g(x) = alpha*f0(x) + beta*f1(x) + gama;   alpha + beta = 1

img1 = cv.imread('./image/vin-logo.png')
img2 = cv.imread('./image/opencv-logo.png')
assert img1 is not None, "file could not be read, check with os.path.exists()"
assert img2 is not None, "file could not be read, check with os.path.exists()"

# Two image should have the same size
img1 = cv.resize(img1, (img2.shape[1], img2.shape[0]))
cv.imshow('img1', img1)
cv.imshow('img2', img2)

# Blending
dst = cv.addWeighted(img1, 0.7, img2, 0.3, 0)
cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()


# ------------------------------- Bitwise Operations -----------------------------------
# If we add 2 images, it will change color. 
# If we blend them, it get a transparent effect.
# We want it to be opaque.
# If it is a rectangular region, we can use ROI.
# But we often work with non-rectangular ROI's, so do it with bitwise operations.

img1 = cv.imread('./image/messi.jpg')
img2 = cv.imread('./image/opencv-logo-white.jpg')
assert img1 is not None, "file could not be read, check with os.path.exists()"
assert img2 is not None, "file could not be read, check with os.path.exists()"
cv.imshow('img1', img1)
cv.imshow('img2', img2)
cv.waitKey(0)

# Want to put logo on top-left corner, so create a ROI
rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]
cv.imshow('roi', roi)
cv.waitKey(0)

# Create a mask of logo and create its inverse mask also
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)   # threshold=10, max=255, pixels in grayscale image < threshold -> set to 0, > threshold -> set to max
mask_inv = cv.bitwise_not(mask)
cv.imshow('img2gray', img2gray)
cv.imshow('mask', mask)
cv.imshow('mask_inv', mask_inv)
cv.waitKey(0)

# Now black-out the area of logo in ROI
img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
cv.imshow('img1_bg', img1_bg)
cv.waitKey(0)

# Take only region of logo from logo image
img2_fg = cv.bitwise_and(img2, img2, mask=mask)
cv.imshow('img2_fg', img2_fg)
cv.waitKey(0)

# Put logo in ROI and modify the main image
dst = cv.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst

cv.imshow('res', img1)
cv.waitKey(0)
cv.destroyAllWindows()