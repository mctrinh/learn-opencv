import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


""" - Find Fourier Transform of images.
    - Utilize FFT functions available in Numpy.
    - Some applications of Fourier Transform.
    Ref: https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html
"""

# ------------------------------- Fourier Transform in Numpy -----------------------------------
''' Fourier Transform is used to analyze the frequency characteristics of various filters.'''

img = cv.imread('./image/messi.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

rows, cols = img.shape
crow,ccol = rows//2 , cols//2
fshift[crow-30:crow+31, ccol-30:ccol+31] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.real(img_back)
plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_back)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
plt.show()


# ------------------------------- Fourier Transform in OpenCV -----------------------------------

img = cv.imread('./image/messi.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

rows, cols = img.shape
crow,ccol = rows/2 , cols/2
# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[int(crow-30):int(crow+30), int(ccol-30):int(ccol+30)] = 1
# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


# ----------------------------- Performance Optimization of DFT --------------------------------
'''
Can modify the size of the array to any optimal size (a product of 2, 3, 5) before finding DFT.
In Numpy, specify the new size of FFT calculation, it will auto pad zeros for you.
In OpenCV, manually pad zeros.
'''
import time

img = cv.imread('./image/messi.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
rows, cols = img.shape
print("{} {}".format(rows,cols))        # 464 824
nrows = cv.getOptimalDFTSize(rows)
ncols = cv.getOptimalDFTSize(cols)
print("{} {}".format(nrows,ncols))      # 480 864

# Pad img with zeros
nimg = np.zeros((nrows,ncols))
nimg[:rows,:cols] = img
plt.subplot(121),plt.imshow(nimg, cmap = 'gray')
plt.title('Numpy padding'), plt.xticks([]), plt.yticks([])

# Pad img with zeros (2nd way)
right = ncols - cols
bottom = nrows - rows
bordertype = cv.BORDER_CONSTANT         # to avoid line breakup in PDF file
nimg = cv.copyMakeBorder(img,0,bottom,0,right,bordertype,value=0)
plt.subplot(122),plt.imshow(nimg, cmap = 'gray')
plt.title('OpenCV padding'), plt.xticks([]), plt.yticks([])
plt.show()

# Calculate DFT performance comparison in Numpy
start = time.time()
fft1 = np.fft.fft2(img)
end = time.time()
print("DFT with Numpy (original shape) takes {} seconds.".format(end-start))
start = time.time()
fft2 = np.fft.fft2(img,[nrows,ncols])
end = time.time()
print("DFT with Numpy (padded shape) takes {} seconds.".format(end-start))

# Calculate DFT performance comparison in OpenCV
start = time.time()
fft1 = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
end = time.time()
print("DFT with OpenCV (original shape) takes {} seconds.".format(end-start))
start = time.time()
fft2 = cv.dft(np.float32(nimg), flags=cv.DFT_COMPLEX_OUTPUT)
end = time.time()
print("DFT with OpenCV (padded shape) takes {} seconds.".format(end-start))

''' --->>> OpenCV is 4x faster than Numpy functions.
    --->>> Processing padded images is faster than original images.'''


# ------------------------ Why Laplacian, Sobel, ... is a High Pass Filter? ---------------------------

# Simple averaging filter without scaling parameter
mean_filter = np.ones((3,3))

# Creating a gaussian filter
x = cv.getGaussianKernel(5,10)
gaussian = x*x.T

# Different edge detecting filters:
# Laplacian
laplacian=np.array([[0, 1, 0], [1,-4, 1], [0, 1, 0]])

# Sobel in x direction
sobel_x= np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# Sobel in y direction
sobel_y= np.array([[-1,-2,-1], [0, 0, 0], [1, 2, 1]])

# Scharr in x-direction
scharr = np.array([[-3, 0, 3], [-10,0,10], [-3, 0, 3]])

filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
filter_name = ['mean_filter', 'gaussian','laplacian', 'sobel_x', 'sobel_y', 'scharr_x']
fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log(np.abs(z)+1) for z in fft_shift]

for i in range(6):
    plt.subplot(2,3,i+1), plt.imshow(mag_spectrum[i],cmap = 'gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])
plt.show()