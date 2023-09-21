import cv2 as cv
import numpy as np


"""
Simple operations based on the image shape. It is normally performed on binary images.
Two inputs: (1) original image, (2) structuring element or kernel which decide the nature 
of operation.
Erosion and Dilation are two basic morphological operators.
"""

# ------------------------------- Erosion --------------------------------------------
'''
The kernel (1) slides through the image, a pixel in the original image is considered 1
only if all the pixels under the kernel is 1, otherwise it is eroded (0)

-> Pixels near boundary will be discarded depending upon the size of kernel.
'''

img = cv.imread('./image/j1.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)
cv.imshow('Erosion', erosion)
cv.waitKey(0)


# ------------------------------- Dilation --------------------------------------------
'''
Opposite of erosion, a pixel is '1' if at least one pixel under the kernel is '1'.
So it increases the white region in the img.

In noise removal, erosion is followed by dilation. Erosion removes white noises, it also
shrinks our object, so we dilate it. Since noise is gone, not come back.

It is useful in joining broken parts of an object.
'''
dilation = cv.dilate(img, kernel, iterations=1)
cv.imshow('Delation', dilation)
cv.waitKey(0)


# --------------------------- Opening = Erosion followed by Dilation ------------------
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
cv.imshow('Opening', opening)
cv.waitKey(0)


# --------------------------- Closing = Dilation followed by Erosion ------------------
''' Useful in closing small holes inside the foreground objects, 
    or small black points on the object.'''
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
cv.imshow('Closing', closing)
cv.waitKey(0)


# ------------------------------------- Morphological Gradient ------------------------
''' It is the difference between dilation and erosion of an image.
    The result will look like the outline of the object.'''
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
cv.imshow('Gradient', gradient)
cv.waitKey(0)


# --------------------------------------- Top Hat -------------------------------------
''' It is the difference between input image and opening of the image.'''
kernel = np.ones((9,9),np.uint8)
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
cv.imshow('Tophat', tophat)
cv.waitKey(0)


# --------------------------------------- Black Hat ----------------------------------
''' It is the difference between input image and closing of the image.'''
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
cv.imshow('Blackhat', blackhat)
cv.waitKey(0)
cv.destroyAllWindows()


# ------------------------------ Various shaped kernels ------------------------------
rectangular_kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
elliptical_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
cross_shaped_kernel = cv.getStructuringElement(cv.MORPH_CROSS,(5,5))

