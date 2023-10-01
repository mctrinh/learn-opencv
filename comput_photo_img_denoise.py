import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# ------------- Remove gaussian noise from color images ----------------
img = cv.imread('./image/gaussian-noise.png')
dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()


# --------------  Remove noise in video -------------------------
cap = cv.VideoCapture('./image/vtest.avi')

# Create a list of first 5 frames
img = [cap.read()[1] for i in range(5)]

# Convert all to grayscale
gray = [cv.cvtColor(i, cv.COLOR_BGR2GRAY) for i in img]

# Convert all to float64
gray = [np.float64(i) for i in gray]

# Create a noise of variance 25
noise = np.random.randn(*gray[1].shape)*10

# Add this noise to images
noisy = [i+noise for i in gray]

# Convert back to uint8
noisy = [np.uint8(np.clip(i,0,255)) for i in noisy]

# Denoise 3rd frame considering all the 5 frames
dst = cv.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)
plt.subplot(131),plt.imshow(gray[2],'gray')
plt.subplot(132),plt.imshow(noisy[2],'gray')
plt.subplot(133),plt.imshow(dst,'gray')
plt.show()
