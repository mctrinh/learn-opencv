# Refs: https://docs.opencv.org/4.x/de/db2/tutorial_py_table_of_contents_histograms.html

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# ------------------------------ Find, Plot, Analyze -----------------------------
"""
Histogram give an overall idea about the intensity distribution of an grayscale image.
X-axis: pixel values [0-255], Y-axis: corresponding number of pixels in the image

Terms:
        BINS: 256 (one for each pixel) [0-255] (histSize in OpenCV)
        DIMS: num of parameters (1: only intensity)
        RANGE: [0,256] the range of intensity values we measure.

        cv.calcHist([img],              <img of type uint8 or float32>
                    [channel],          <[0] for grayscale img, [0] [1] or [2] for B, G, R channel
                    mask,               <None: full image>
                    histSize,           <[256] for full scale
                    ranges[, hist[, accumulate]])   <Normally [0,256]
"""

img = cv.imread('./image/home.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()


# cv.calcHist() is 40x faster than np.histogram()
hist = cv.calcHist([img], [0], None, [256], [0,256])


# For 1D histogram, hist = np.bincount(img.ravel(), minlength=256) is 10x faster than np.histogram()
hist, bins = np.histogram(img.ravel(), 256, [0,256])    # bins have 257 elements, as np calculates bins as 0-00.99, ..., 255-255.99


# Matplotlib can directly find and plot the histogram
plt.hist(img.ravel(),256,[0,256])
plt.show()


# Normal plot of matplotlib
img = cv.imread('./image/home.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr, color=col)
    plt.xlim([0,256])
plt.show()


# ------------ Using OpenCV to plot ---------------
bins = np.arange(256).reshape(256,1)

def hist_curve(im):
    h = np.zeros((300,256,3))
    if len(im.shape) == 2:
        color = [(255,255,255)]
    elif im.shape[2] == 3:
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, col in enumerate(color):
        hist_item = cv.calcHist([im],[ch],None,[256],[0,255])
        cv.normalize(hist_item,hist_item,0,255,cv.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins,hist)))
        cv.polylines(h,[pts],False,col)
    y=np.flipud(h)
    return y

def hist_lines(im):
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print("hist_lines applicable only for grayscale images")
        #print "so converting image to grayscale for representation"
        im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
    hist_item = cv.calcHist([im],[0],None,[256],[0,255])
    cv.normalize(hist_item,hist_item,0,255,cv.NORM_MINMAX)
    hist = np.int32(np.around(hist_item))
    for x,y in enumerate(hist):
        cv.line(h,(int(x),0),(int(x),int(y)),(255,255,255),5)
    y = np.flipud(h)
    return y

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Show histogram for color image in curve mode
curve = hist_curve(img)
cv.imshow('Curve mode',curve)
cv.imshow('Color image',img)
cv.waitKey(0)

# Show histogram in bin mode
lines = hist_lines(img)
cv.imshow('Bin mode',lines)
cv.imshow('Gray image',gray)
cv.waitKey(0)

# Show equalized histogram (always in bin mode)
equ = cv.equalizeHist(gray)
lines = hist_lines(equ)
cv.imshow('Histogram lines',lines)
cv.imshow('Equilized histogram',equ)
cv.waitKey(0)

# Show histogram for gray image in curve mode
curve = hist_curve(gray)
cv.imshow('Curve mode 1',curve)
cv.imshow('Gray image 1',gray)
cv.waitKey(0)

# Show histogram for a normalized image in curve mode
norm = cv.normalize(gray, gray, alpha = 0,beta = 255,norm_type = cv.NORM_MINMAX)
lines = hist_lines(norm)
cv.imshow('Histogram lines 1',lines)
cv.imshow('Normalized image',norm)
cv.waitKey(0)
cv.destroyAllWindows()




# ------------------------------ Histogram Equalization -----------------------------
"""
Used as a "reference tool" to make all images with same lighting conditions.
This is useful in many cases. For example, in face recognition, before training
the face data, the images of faces are histogram equalized to make them all
with same lighting conditions.

Histogram equalization is good when histogram of the image is confined to a
particular region. It won't work good in places where there is large intensity
variations where histogram covers a large region, ie both bright and dark pixels
are present.
"""

img = cv.imread('./image/wiki.jpeg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]

hist,bins = np.histogram(img2.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img2.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()


# Global Histograms Equalization in OpenCV
img = cv.imread('./image/wiki.jpeg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
equ = cv.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv.imwrite('./image/wiki-equalized.png',res)


# CLAHE: Contrast Limited Adaptive Histogram Equalization
# (tilesize=8x8, each tile is histogram equalized as usual)
''' If there is noise, it will be amplified. Apply constrast limiting to avoid it.
    If any histogram bin is above the specified contrast limit, default=40, those
    pixels are clipped and distributed uniformly to other bins before applying
    histogram equalization. After equalization, to remove artifacts in title borders,
    bilinear interpolation is applied.
'''

img = cv.imread('./image/tomato.jpeg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

''' Global histogram equalization'''
equ = cv.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv.imwrite('./image/tomato-equalized.png',res)

# Create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
cv.imwrite('./image/tomato-clahe.png',cl1)




# ------------------------------ 2D Histograms -----------------------------
"""
1D histogram: 1 feature (grayscale intensity value of pixels), BGR -> grayscale.
2D histogram: 2 features (hue & saturation values of pixels),  BGR -> HSV (color histogram)

cv.calcHist(channels=[0,1],             : both hue and saturation plane
            bins=[180,256],
            range=[0,180,0,255])
"""

img = cv.imread('./image/home-1.jpeg')
assert img is not None, "file could not be read, check with os.path.exists()"
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

# Plot a grayscale histogram
cv.imshow("2D Histogram", hist)
cv.waitKey(0)

# Plot using matplotlib
plt.imshow(hist, interpolation='nearest')
plt.show()



# ------------------------------ Histogram Backprojection -----------------------------
"""
Refs: https://docs.opencv.org/4.8.0/dc/df6/tutorial_py_histogram_backprojection.html

What is it actually in simple words?
It is used for image segmentation or finding objects of interest in an image.
It creates an image of the same size (but single channel) as that of our input image,
where each pixel corresponds to the probability of that pixel belonging to our object.
The output image will have our object of interest in more white compared to remaining part.
Histogram Backprojection is used with camshift algorithm etc.

How do we do it?
We create a histogram of an image containing our object of interest (in our case, the ground,
leaving player and other things). The object should fill the image as far as possible for
better results. And a color histogram is preferred over grayscale histogram, because
color of the object is a better way to define the object than its grayscale intensity.
We then "back-project" this histogram over our test image where we need to find the object,
we calculate the probability of every pixel belonging to the ground and show it.
The resulting output on proper thresholding gives us the ground alone.
"""


# ------  Numpy ------
# Roi is the object or region of object we need to find
img = cv.imread('./image/messi.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
roi = img[380:460, 50:130]
hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)

# Target is the image we search in
target = cv.imread('./image/messi.jpg')
assert target is not None, "file could not be read, check with os.path.exists()"
hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)

# Find the histograms using calcHist. Can be done with np.histogram2d also
M = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256])
I = cv.calcHist([hsvt],[0, 1], None, [180, 256], [0, 180, 0, 256])

'''
Ratio R=M/I, then backproject R, use R as palette and create a new image with every pixel
as its corresponding probability of being target B(x,y) = R[h(x,y), s(x,y)], where h is
hue and s is saturation of the pixel at (x,y).
Then apply the condition B(x,y) = min[B(x,y),1]
'''
R = np.divide(M,I,out=np.zeros_like(M),where=I!=0)
h,s,v = cv.split(hsvt)
B = R[h.ravel(),s.ravel()]
B = np.minimum(B,1)
B = B.reshape(hsvt.shape[:2])

'''
Now apply a convolution with a circular disc, B = D*B, where D is the disc kernel
'''
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
cv.filter2D(B,-1,disc,B)
B = np.uint8(B)
cv.normalize(B,B,0,255,cv.NORM_MINMAX)

'''
Now the location of maximum intensity gives us the location of object.
If we expect a region in the image, thresholding for a suitable value gives a nice result.
'''
ret,thresh = cv.threshold(B,50,255,0)


# ------ Backprojection in OpenCV ------
img = cv.imread('./image/messi.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
roi = img[380:460, 50:130]
hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)

target = cv.imread('./image/messi.jpg')
assert target is not None, "file could not be read, check with os.path.exists()"
hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)

# Calculating object histogram
roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )

# Normalize histogram and apply backprojection
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
dst = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)

# Now convolute with circular disc
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
cv.filter2D(dst,-1,disc,dst)

# Threshold and binary AND
ret,thresh = cv.threshold(dst,50,255,0)
thresh = cv.merge((thresh,thresh,thresh))
res = cv.bitwise_and(target,thresh)
res = np.vstack((target,thresh,res))
cv.imwrite('./image/messi-backprojection.jpg',res)
