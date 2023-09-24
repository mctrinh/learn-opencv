import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


""" Use GrabCut algorithm to extract foreground in images."""
# https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html

'''
- Draw a rectangle around the foreground region.
- Algorithm segments it iteratively to get the best result.
- Draw white strokes (denoting foreground) and black strokes (denoting background)
'''

img = cv.imread('./image/messi.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (0,0,820,460)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img),plt.colorbar(),plt.show()


# Newmask is the mask image I manually labelled
newmask = cv.imread('./image/messi-mask.jpg', cv.IMREAD_GRAYSCALE)
assert newmask is not None, "file could not be read, check with os.path.exists()"
# Wherever it is marked white (sure foreground), change mask=1
# Wherever it is marked black (sure background), change mask=0
mask[newmask == 0] = 0
mask[newmask == 255] = 1
mask, bgdModel, fgdModel = cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()