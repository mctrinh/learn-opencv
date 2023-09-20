import cv2 as cv
import numpy as np


# ------------------------------- Image Addition ------------------------------
# cv.add() to add two images (saturated operation)
# or simply by the numpy operation res = img1 + img2 (modulo operation)
# both images should be of same depth, or 2nd image can be a scalar value.

x = np.uint8([250])
y = np.uint8([10])
print(cv.add(x,y))  # 250+10=260 -> 255

print(x+y)  # 250+10=260 % 256 = 4 -> cd.add() provide a better result.


# ------------------------------- Image Blending ------------------------------
