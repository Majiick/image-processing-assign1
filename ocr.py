import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('text.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original gray', gray_img)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray_img)
cv2.imshow('Clahe equalized gray', cl1)

cl1 = cv2.bilateralFilter(cl1, 9, 75, 75)
thresholded_img = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
#ret, thresholded_img = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('Clahe processed img', thresholded_img)


histogram_equalized = cv2.equalizeHist(gray_img.ravel())
plt.hist(histogram_equalized, 256, [0,256])

equalized_img = histogram_equalized.reshape(gray_img.shape)
cv2.imshow('Equalized gray', equalized_img)

equalized_img = cv2.bilateralFilter(equalized_img, 9, 75, 75)
thresholded_img = cv2.adaptiveThreshold(equalized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 6)
cv2.imshow('Equalized processed img', thresholded_img)


#plt.show()
while True:
    if cv2.waitKey(33) == ord('q'):
        cv2.destroyAllWindows()
        exit()