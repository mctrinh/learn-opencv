# https://johnwlambert.github.io/stereo/
# https://docs.opencv.org/4.8.0/dd/d53/tutorial_py_depthmap.html

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


imgL = cv.imread('./image/tsukuba_l.png', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('./image/tsukuba_r.png', cv.IMREAD_GRAYSCALE)

#stereo = cv.StereoBM.create(numDisparities=16, blockSize=15)
stereo = cv.StereoSGBM.create(minDisparity=0,
                              numDisparities=16,
                              blockSize=15,
                              P1=600,
                              P2=2400,
                              disp12MaxDiff=20,
                              preFilterCap=16,
                              uniquenessRatio=1,
                              speckleWindowSize=100,
                              speckleRange=20,
                              mode=1
                              )
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()
#disparity = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,dtype=cv.CV_8U)
#cv.imshow('disparity', disparity)
#cv.waitKey(0)
#cv.destroyAllWindows()
