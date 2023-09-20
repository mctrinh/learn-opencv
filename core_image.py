""" Access pixel values and modify them
    Access image properties
    Set a Region of Interest (ROI)
    Split and merge images
"""

import cv2 as cv
import numpy as np


img = cv.imread('./image/messi.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"


# ------------------------------- Accessing and Modifying pixel values ------------------------------

# Access a pixel values (BGR values for a BGR image, or intensity for a grayscale image)
px = img[100,100] 
print(px)           

# Access only blue pixel
blue = img[100,100,0]
print(blue)

# Modify the pixel values the same way
img[100,100]  = [255,255,255]
print(img[100,100])

''' Numpy is optimized for fast array calculations.
    So accessing each pixel value and modifying it will be very slow, discouraged.
    Above method is good for selecting a region of an array.
    
    For individual pixel access, use array.item(), array.itemset()
    They return a scalar, so to access all B,G,R values, need to call array.item() separately for each value.'''

# Access RED value
red = img.item(10,10,2)
print(red)

# Modify RED value
img.itemset((10,10,2),100)
red = img.item(10,10,2)
print(red)


# ------------------------------- Accessing Image Properties ------------------------------

# Shape of an image, return a tuple (# of rows = HEIGHT, columns = WIDTH, channels)
# For an gray image, only return (# of rows, colums)
print(img.shape)

# Total numner of pixels
print(img.size)

# Image datatype (important while debugging, many errors caused by invalid datatype)
print(img.dtype)


# ------------------------------- Image ROI ------------------------------
''' If need to detect eye, first perform face detection over the entire image, 
    select the face region alone, search for eyes inside the face region.
    It improve the accuracy and performance.'''

# Select the ball and copy it to another region in the image
ball = img[380:460, 520:600]
img[380:460, 620:700] = ball

cv.imshow('image', img)
cv.imshow('ball', ball)
cv.waitKey(0)


# ------------------------------- Splitting and Merging Image Channels ------------------------------
