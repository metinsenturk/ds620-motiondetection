import numpy as np
import cv2 as cv

img = cv.imread('files/messi5.jpg')

pxl = img[100, 100]
print(pxl)

blue = img[100, 100, 0]
print(blue)

img[100, 100] = [255, 255, 255]
print(img)
print(img[100, 100])

print(img.shape)
print(img.size)
print(img.dtype)

ball = img[280:340, 330:390]
img[273:333, 100:160] = ball

b, g, r = cv.split(img)

img = cv.merge((b, g, r))

cv.imshow('res',img)
cv.waitKey(0)
cv.destroyAllWindows()