import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------- 2D Convolution (Image Filtering) -----------------------------------
"""
As in 1D signals, images can be filtered with various low-pass filters (LPF), high-pass filters (HPF), ...
LPF helps in removing noise, HPF helps in finding edges in images.

cv.filter2D() convolves a kernel with an image. A 5x5 averaging filter kernel will look like:
                            |1 1 1 1 1|
                            |1 1 1 1 1|
                 K = 1/25 * |1 1 1 1 1|
                            |1 1 1 1 1|
                            |1 1 1 1 1|
"""

img = cv.imread('./image/opencv-logo.png')
assert img is not None, "file could not be read, check with os.path.exists()"

kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
# plt.show()


# ------------------------------- Image Blurring (Image Smoothing) -----------------------------------
"""
Convolve the image with a LPF kernel to remove noise, it actually removes high frequency content 
(e.g., noise, edges), thus edges are blurred a little bit in this operation.

Four main types of blurring techniques:
        Averaging, Gaussian Blurring, Median Blurring, Bilateral Filtering
"""

# Averaging
'''
A 3x3 normalized box filter look like:
                            |1 1 1|
                  K = 1/9 * |1 1 1|
                            |1 1 1|
'''
img = cv.imread('./image/opencv-logo-white.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
blur = cv.blur(img,(5,5))
plt.subplot(221),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(blur),plt.title('Averaging Blurred')
plt.xticks([]), plt.yticks([])

# Gaussian Blurring
blur = cv.GaussianBlur(img,(5,5),0)
plt.subplot(223),plt.imshow(blur),plt.title('Gaussian Blurred')
plt.xticks([]), plt.yticks([])
# plt.show()


# Median Blurring
'''
Highly effective against salt-and-pepper noise in an image.
'''
img = cv.imread('./image/s&p4.png')
assert img is not None, "file could not be read, check with os.path.exists()"
cv.imshow('Salt & Pepper noise', img)
cv.waitKey(0)
median = cv.medianBlur(img,5)
cv.imshow('medianBlur', median)
cv.waitKey(0)


# Bilateral Fitering
'''
Highly effective in noise removal while keeping edges sharp, operation is slower compared to other filters.
'''
img = cv.imread('./image/texture.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
cv.imshow('Texture', img)
cv.waitKey(0)
blur = cv.bilateralFilter(img,9,75,75)
cv.imshow('Bilateral Fitering', blur)
cv.waitKey(0)
cv.destroyAllWindows()