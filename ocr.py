import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('text.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original gray', gray_img)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray_img)

cl2 = cv2.bilateralFilter(cl1, 9, 75, 75)
thresholded_img = cv2.adaptiveThreshold(cl2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
#ret, thresholded_img = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

top_pixel = float('inf')
bottom_pixel = 0
right_pixel = 0
left_pixel = float('inf')

for y, r in enumerate(thresholded_img):
    for x, c in enumerate(r):
        if thresholded_img[y, x] == 0:
            if y < top_pixel:
                top_pixel = y
            if y > bottom_pixel:
                bottom_pixel = y
            if x > right_pixel:
                right_pixel = x
            if x < left_pixel:
                left_pixel = x

cv2.rectangle(thresholded_img,(left_pixel, top_pixel),(right_pixel,bottom_pixel),(0,255,0),3)
cv2.imshow('Clahe processed img', thresholded_img)

cl1 = cl1[top_pixel-5:top_pixel+(bottom_pixel-top_pixel)+5, left_pixel-5:left_pixel+(right_pixel-left_pixel)+5]
edges = cv2.Canny(cl1,100,200)
cv2.imshow('Edges',edges)
# https://stackoverflow.com/questions/44047819/increase-image-brightness-without-overflow/44054699#44054699
dilated_img = cv2.dilate(cl1, np.ones((7,7), np.uint8)) 
bg_img = cv2.medianBlur(dilated_img, 21)
diff_img = 255 - cv2.absdiff(cl1, bg_img)
norm_img = diff_img.copy() # Needed for 3.x compatibility
cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
cv2.imshow('test', norm_img)
cv2.imshow('Clahe equalized gray', cl1)


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