# Refs: https://docs.opencv.org/4.x/d3/d05/tutorial_py_table_of_contents_contours.html

import numpy as np
import cv2 as cv


# --------------------------------------- Draw Contours ----------------------------------

"""
Contour: 

A curve joining all continuous points along the boundary, having same color
or intensity. It is useful for shape analysis, object detection and recognition.

For better accuracy, use binary images.
Before finding contours, apply threshold or canny edge detection.

Objects to be found should be white and background should be black.
"""

img = cv.imread('./image/H.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
ret, thresh = cv.threshold(img, 127, 255, 0)

# contours: a list of all contours in the image
# 2nd arg: a contour retrieval mode
# 3rd arg: a contour approximation method, cv.CHAIN_APPROX_NONE: store all boundary points
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

# Draw contours use cv.drawContours(img, contours, index of contours {-1 for all}, color, thickness, ...)
cv.drawContours(img, contours, -1, (0,255,0), 3)

# Draw an individual contour, e.x., 1st contour
cv.drawContours(img, contours, 0, (0,255,0), 3)

# Useful manner
cnt = contours[0]
cv.drawContours(img, [cnt], 0, (0,255,0), 3)

cv.imshow('Image with contours', img)
cv.waitKey(0)
cv.destroyAllWindows()


# --------------------------------------- Contour Features: area, perimeter, centroid, ... ----------------------------------
# Momments
''' Calculate center of mass of the object, area of the object, ...
    cv.moments() gives a dic of all moment values calculated
'''

img = cv.imread('./image/star.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
ret, thresh = cv.threshold(img, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv.moments(cnt)
print(M)


# Centroid
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])


# Contour area
area = cv.contourArea(cnt)
area = int(M['m00'])            # alternative


# Contour perimeter
perimeter = cv.arcLength(cnt,True)     # True: a closed contour, False: a curve


# Contour approximation
''' Approximate a contour shape to another shape with less number of vertices, depending upon the precision we specify.
    It use Douglas Peucker algorithm, epsilon is a accuracy parameter.
    cv.approxPolyDP() returns coordinates of approximate points.
'''
img = cv.imread('./image/rectangle-bad.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
ret, thresh = cv.threshold(img, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]
img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
cv.drawContours(img_bgr, [cnt], 0, (0,255,0), 3)
cv.imshow('Image with contours', img_bgr)
cv.waitKey(0)

epsilon = 0.05 * cv.arcLength(cnt,True)             # 0.05 give a rectangle, 0.0001 give exact like contours
approx = cv.approxPolyDP(cnt, epsilon, True)
img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
cv.drawContours(img_bgr, [approx], 0, (0,255,0), 3)
cv.imshow('Image with approximate contours', img_bgr)
cv.waitKey(0)
cv.destroyAllWindows()


# Convex Hull
'''
cv.convexHull() checks a curve for convexity defects (bulged inside) and corrects it. 
hull = cv.convexHull(points: contour points coords,
                     hull: output (avoid it),
                     clockwise: True/False
                     returnPoints: True (hull points coords) / False (indices of contour points)
'''
img = cv.imread('./image/hand.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
ret, thresh = cv.threshold(img, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]
hull = cv.convexHull(cnt)
img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
cv.drawContours(img_bgr, [hull], 0, (0,255,0), 3)
cv.imshow('Image with hull', img_bgr)
cv.waitKey(0)
cv.destroyAllWindows()


# Check convexity of a curve
print(cv.isContourConvex(cnt))
print(cv.isContourConvex(hull))


# Straight bounding rectangle
'''Do not consider the rotation of the object, thus area of bounding rectangle won't be minimum.'''
img_gray = cv.imread('./image/lightning.jpg', cv.IMREAD_GRAYSCALE)
assert img_gray is not None, "file could not be read, check with os.path.exists()"
ret, thresh = cv.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]
x,y,w,h = cv.boundingRect(cnt)
img = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)
cv.imshow('Bounding rectangle', img)
cv.waitKey(0)


# Rotated bounding rectangle
''' Consider the rotation, bounding rectangle is drawn with minimum area.'''
rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(img, [box], 0, (0,0,255), 2)
cv.imshow("Bounding rectangle", img)
cv.waitKey(0)


# Minimum enclosing circle
''' Find the circumcircle of an object'''
(x,y), radius = cv.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
cv.circle(img, center, radius, (255,0,0), 2)
cv.imshow("Bounding rectangle", img)
cv.waitKey(0)


# Fitting an Ellipse
''' Fit an ellipse to an object, returns the rotated rectangle in which the ellipse is inscribed.'''
ellipse = cv.fitEllipse(cnt)
cv.ellipse(img, ellipse, (0, 255, 255), 2)
cv.imshow("Bounding rectangle", img)
cv.waitKey(0)


# Fitting a line
''' Fit a line to a set of points'''
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv.fitLine(cnt,cv.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv.line(img,(cols-1,righty),(0,lefty),(255,0,255),2)
cv.imshow("Bounding rectangle", img)
cv.waitKey(0)
cv.destroyAllWindows()



# --------------- Contour Properties: aspect ratio, extent, solidity, equavalent diameter, orientation, ... ---------------

# Aspect ratio = width / height of bounding rect of the object
x,y,w,h = cv.boundingRect(cnt)
aspect_ratio = float(w)/h
print("Aspect ratio: ", aspect_ratio)


# Extent = object area / bounding rectangle area
area = cv.contourArea(cnt)
rect_area = w*h
extent = float(area)/rect_area
print("Extent: ", extent)


# Solidity = contour area / convex hull area
hull = cv.convexHull(cnt)
hull_rea = cv.contourArea(hull)
solidity = float(area) / hull_rea
print("Solidity: ", solidity)


# Equivalent diameter = sqrt(4*contour_area/pi)
''' The diameter of the circle whose area is same as the contour area.'''
equi_diameter = np.sqrt(4*area/np.pi)
print("Equivalent diameter: ", equi_diameter)


# Orientation
''' The angle at which the object is directed.'''
(x,y), (MA,ma), angle = cv.fitEllipse(cnt)          # position, lengths of Major Axis and minor axis, angle
print("Angle: ", angle)


# Mask and pixel points
''' Extract all the points comprising the object.'''
mask = np.zeros(img_gray.shape, np.uint8)
cv.drawContours(mask, [cnt], 0, 255, -1)
pixelpoints = np.transpose(np.nonzero(mask))
pixelpoints = cv.findNonZero(mask)              # alternative


# Max value, min value, and their locations
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(img_gray, mask=mask)


# Mean color (BGR) or mean intensity (grayscale)
mean_val = cv.mean(img_gray, mask=mask)


# Extreme points
''' Topmost, bottommost, rightmost, leftmost points of the object'''
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

for point in [leftmost,rightmost,topmost,bottommost]:
    img_extreme = cv.circle(img, point, 15, (255,255,0), -1)
    cv.imshow("Extreme points", img_extreme)
    cv.waitKey(0)

cv.destroyAllWindows()



# --------------------------------------- Contour More Functions -------------------------------------
'''
- Convexity defects and how to find them
- Find the shortest distance form a point to a polygon
- Matching different shapes '''

# Convexity defects

img = cv.imread('./image/star1.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret,thresh = cv.threshold(img_gray, 127, 255,0)
contours,hierarchy = cv.findContours(thresh,2,1)
cnt = contours[0]

hull = cv.convexHull(cnt,returnPoints = False)
defects = cv.convexityDefects(cnt,hull)
print("hull", hull)
print("defects", defects)
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv.line(img,start,end,[0,255,0],2)
    cv.circle(img,far,5,[0,0,255],-1)
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()


# Point Polygon Test
''' Find the shortest distance between a point in the image and a contour.
    The distance is negative when point is outside the contour, positive
    when point is inside, zero if point is on the contour.

    Args:
            cnt: contours
            (,): point coords
            True: find the signed distance
            False: returns +1, -1, 0 (inside, outside, on the contour), 2x-3x speed
'''
dist = cv.pointPolygonTest(cnt, (50,50), True)
print("Dist", dist)


# Match Shapes
''' Compare 2 shapes, 2 contours, returns a metric showing the similarity.
    The lower the result, the better match it is.'''
img1 = cv.imread('./image/star1.jpg', cv.IMREAD_GRAYSCALE)
# img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('./image/star2.jpg', cv.IMREAD_GRAYSCALE)
# img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
img3 = cv.imread('./image/star3.jpg', cv.IMREAD_GRAYSCALE)
# img3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
assert img1 is not None, "file could not be read, check with os.path.exists()"
assert img2 is not None, "file could not be read, check with os.path.exists()"
assert img3 is not None, "file could not be read, check with os.path.exists()"

ret1, thresh1 = cv.threshold(img1,127,255,0)
contours, hierarchy = cv.findContours(thresh1,2,1)
cnt1 = contours[0]

ret2, thresh2 = cv.threshold(img2,127,255,0)
contours, hierarchy = cv.findContours(thresh2,2,1)
cnt2 = contours[0]

ret3, thresh3 = cv.threshold(img3,127,255,0)
contours, hierarchy = cv.findContours(thresh3,2,1)
cnt3 = contours[0]

ret11 = cv.matchShapes(cnt1,cnt1,1,0.0)
ret12 = cv.matchShapes(cnt1,cnt2,1,0.0)
ret13 = cv.matchShapes(cnt1,cnt3,1,0.0)
print("Matching Image 1 with itself", ret11)
print("Matching Image 1 with Image 2", ret12)
print("Matching Image 1 with Image 3", ret13)


# --------------------------------------- Contour Hierarchy -------------------------------------
''' Parent-child relationship in contours.'''