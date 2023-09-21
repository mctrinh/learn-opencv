# A small app to find the Canny edge detection whose threshold values can be varied using two trackbars.

import cv2 as cv


# Create a black image, a window
img = cv.imread('./image/messi.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv.Canny(img, 0, 100)

cv.namedWindow('edgeDetector')

def nothing(x):
    pass

# Create trackbars for threshold change
# args: trackbar name, window name, default value, max value, callback function that is executed every time trackbar value change.
cv.createTrackbar('minVal', 'edgeDetector', 0, 300, nothing)
cv.createTrackbar('maxVal', 'edgeDetector', 0, 300, nothing)

while(1):
    cv.imshow('edgeDetector', edges)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # get current positions of four trackbars
    minVal = cv.getTrackbarPos('minVal', 'edgeDetector')
    maxVal = cv.getTrackbarPos('maxVal', 'edgeDetector')
    edges = cv.Canny(img, minVal, maxVal)

cv.destroyAllWindows()
