import numpy as np
import cv2 as cv

img = cv.imread('./image/florence.jpeg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp = sift.detect(gray,None)

img = cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # flags is optional

cv.imwrite('./image/florence_sift_keypoints.jpg',img)

# Calculate the descriptor: 1st method
kp,des = sift.compute(gray,kp)

# Calculate the descriptor: 2nd method
# kp: a list of keypoints, des: an array of shape no. of keypoints x 128
sift = cv.SIFT_create()
kp,des = sift.detectAndCompute(gray,None)
