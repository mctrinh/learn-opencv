import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


img = cv.imread('./image/fly.jpeg',cv.COLOR_BGR2GRAY)

# Create a SURF object
surf = cv.xfeatures2d.SURF_create(400)  # Hessian threshold = 400

# Find keypoints and descriptors directly
kp,des = surf.detectAndCompute(img,None)
print(len(kp))  # reduce it to around 50 for drawing

# Check present Hessian threshold
print(surf.getHessianThreshold())

# Set it to 50000, it is just for representing in picture.
# In actual cases, it is better to have a value 300-500
surf.setHessianThreshold(50000)

# Again compute keypoints and check its number
kp,des = surf.detectAndCompute(img,None)
print(len(kp))

# If less than 50, let's draw it on the image
img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2), plt.show()

# Check upright flag, if it False, set it to True
print(surf.getUpright())
surf.setUpright(True)

# Recompute the feature points and draw it
kp = surf.detect(img,None)
img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)

plt.imshow(img2),plt.show()

# Find size of descriptor
print(surf.descriptorSize())

# That means flag, "extended" is False.
surf.getExtended()

# So we make it to True to get 128-dim descriptors.
surf.setExtended(True)
kp, des = surf.detectAndCompute(img,None)
print(surf.descriptorSize())
print(des.shape)
