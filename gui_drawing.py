import cv2 as cv
import numpy as np


# Create a black image
img = np.zeros((512,512,3), np.uint8)

# Draw a diagonal blue line with thickness of 5px
cv.line(img, (0,0), (511,511), (255,0,0), 5)

# Draw a rectangle with points at top-left and bottom-right corners, thickness 3px
cv.rectangle(img, (384,0), (510,128), (0,255,0), 3)

# Draw a circle with its center coordinates and radius, -1 to fill it, default thickness 1px
cv.circle(img, (447,63), 63, (0,0,255), -1)     # (img, center, radius, color, fill)

# Draw a half ellipse at the center of the image
cv.ellipse(img, (256,256), (100,50), 0, 0, 180, 255, -1)        # color 255 == (255,0,0)

# Draw a polygon with four vertices in yellow color
pts = np.array([[10,5], [20,30], [70,20], [50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img, [pts], True, (0,255,255))

# Adding text to image
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, 'OpenCV', (10,500), font, 4, (255,255,255), 2, cv.LINE_AA)

cv.imshow("Display window", img)
k = cv.waitKey(0)
if k == ord("s"):
    cv.imwrite("./image/drawing.png", img)
