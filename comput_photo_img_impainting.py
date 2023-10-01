# https://docs.opencv.org/4.8.0/df/d3d/tutorial_py_inpainting.html

import numpy as np
import cv2 as cv


img = cv.imread('./image/messi-3.jpeg')
mask = cv.imread('./image/messi-3-mask.jpeg', cv.IMREAD_GRAYSCALE)
dst = cv.inpaint(img,mask,3,cv.INPAINT_TELEA)
cv.imshow('dst',dst)
cv.waitKey(0)
cv.destroyAllWindows()
