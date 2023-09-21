import cv2 as cv
import numpy as np

''' The code should provide the correct solution and in the fastest manner.'''


# ------------------------------- Measuring Performance with OpenCV -----------------------------------
e1 = cv.getTickCount()
print("We are measuring the time taken by the print() to print this line!")
e2 = cv.getTickCount()
time = (e2-e1)/cv.getTickFrequency()
print(f"The print function takes: {time} sec")

# Apply median filtering with kernels of odd sizes ranging from 5 to 21
img1 = cv.imread('./image/messi.jpg')
assert img1 is not None, "file could not be read, check with os.path.exists()"
e1 = cv.getTickCount()
for i in range(5,21,2):
    img1 = cv.medianBlur(img1,i)
e2 = cv.getTickCount()
t = (e2-e1)/cv.getTickFrequency()
print("The median filtering process takes: {} sec".format(t))
cv.imshow('img1', img1)
cv.waitKey(0)
cv.destroyAllWindows()


# ------------------------------- Default Optimization in OpenCV -------------------------------------
# cv.useOptimized() checks whether optimized or unoptimized code is run (optimization is enable or not)
# cv.setUseOptimized() to enable/disable it.

# Optimization is enabled by default
print("Is optimization enabled?", cv.useOptimized())

# Disable it
# cv.setUseOptimized(False)
print("Is optimization enabled?", cv.useOptimized())

# Python scalar operations are faster than Numpy scalar operations, Numpy shows advantages in bigger arrays.
x = 5
e1 = cv.getTickCount()
y = x**2
e2 = cv.getTickCount()
t = (e2-e1)/cv.getTickFrequency()
print("y=x**2 takes: {} sec".format(t))
e1 = cv.getTickCount()
y = x*x
e2 = cv.getTickCount()
t = (e2-e1)/cv.getTickFrequency()
print("y=x*x takes: {} sec".format(t))

z = np.uint8([5])
e1 = cv.getTickCount()
y = z*z
e2 = cv.getTickCount()
t = (e2-e1)/cv.getTickFrequency()
print("y=z*z takes: {} sec".format(t))
e1 = cv.getTickCount()
y = np.square(z)
e2 = cv.getTickCount()
t = (e2-e1)/cv.getTickFrequency()
print("y=np.square(z) takes: {} sec".format(t))

img1gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
e1 = cv.getTickCount()
z = cv.countNonZero(img1gray)   # take 1-channel image only (grayscale)
e2 = cv.getTickCount()
t = (e2-e1)/cv.getTickFrequency()
print("cv.countNonZero() takes: {} sec".format(t))
e1 = cv.getTickCount()
z = np.count_nonzero(img1gray)   # take 1-channel image only (grayscale)
e2 = cv.getTickCount()
t = (e2-e1)/cv.getTickFrequency()
print("np.count_nonzero() takes: {} sec".format(t))


'''
To make code faster:
    - Avoid using loops in Python as much as possible, especially double/triple loops. They are inherently slow.
    - Vectorize code to the maximum extent possible, as Numpy and OpenCV are optimized for vector operations.
    - Exploit the cache coherence.
    - Never make copies of an array unless it is necessary. Try to use views instead. Array copying is a costly operation.
Code is still slow after doing these operations, use additional libraries like Cython to make it faster.

'''
