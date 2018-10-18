# Zan Smirnov C15437072 Image Processing Assignment 1
# This script takes in an image of text, for example a poor quality Russian apartment sale contract
# and tries to improve the quality of the image as to make the text more readable.

# The general method I use is dilating and blurring the image and getting the difference between
# the blurred image and the dilated image to make the text stand out and cancel out the
# background. Just using thresholding such as Clahe works well too but these two methods
# in combination provide the best result. Clahe fixes the lighting and the subtraction removes further lighting issues and noise.

# To crop the image we blur it, threshold it and get the topmost, bottommost, rightmost, and leftmost black
# text pixels to use as the cropping rectangle.

# First we convert the image to grayscale.
# I chose to convert the image to grayscale instead of using the L channel of the LAB color space 
# because the results were the same but using grayscale is easier.

# Then we equalize the histogram to minimize the lighting issues.
# Clahe here is preferred over normal equalization because it it adaptive.
# This means that instead of equalizing the global histogram, clahe equalizes local areas 
# that are the size of the kernel which in this case is 8,8.
# Adaptive thresholding works well in our test image because the lighting is not even and
# changes a lot from area to area of the image.

# After CLAHE I apply smoothing bilateral filter in order to get rid of some of the noisy black spots that isn't text.
# The bilateral filter basically dissolves the black noise in the background into the background.
# This denoised blurred image is then thresholded and used to acquire the topmost, bottommost, rightmost, and leftmost black pixels for cropping.

# After cropping the image, we then apply our dilation, blurring and difference between the original clahe'd image and the dilated image.
# When we dilate and blur we're left with a blurry image that has the background of the image without the text.
# The text dissolves into the background basically.
# Now we get the difference between the original clahe'd image and the blurred image of the background.
# This means that the difference between the text pixels will be great because in the original image
# the text is very black but in the blurred image it is blurred with the background and takes the color of the background
# which isn't as black.
# The difference between the background pixels however will not be as great because they both will be about the same value.
# This means that the text will stand out while the background will be subtracted.

# After we got the image of the difference between the background blurred image and the original image,
# we normalize the image to increase the contrast and make the text stand out even more.
# Then we display the image.

# I use the method for dilating, blurring and canceling out the background in the SO answer below.
# Masek, D. (2017). Increase image brightness without overflow. [online] stackoverflow.com. Available at: https://stackoverflow.com/questions/44047819/increase-image-brightness-without-overflow/44054699#44054699 [Accessed 18 Oct. 2018].

import cv2
import numpy as np
from matplotlib import pyplot as plt
import easygui

# Make the user open the image
print("Please open the target image")
f = easygui.fileopenbox()
img = cv2.imread(f)

# Convert the image to gray
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# Apply the clahe equalization
cl1 = clahe.apply(gray_img)
cl2 = cv2.bilateralFilter(cl1, 9, 75, 75)
# Now we threshold the image, again using adaptive thresholding in order to make all the text black
# and make all of the background white. While we did apply clahe, there are still small differences in lighting
# in the image. The kernel size is 3 because the letters are small.
thresholded_img = cv2.adaptiveThreshold(cl2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)


# Now we use the thresholded image to figure out how to crop the image.
# We do this by getting the location of the topmost, bottommost, rightmost, and leftmost black pixel.
# In this case the black pixels are the text.
top_pixel = float('inf')
bottom_pixel = 0
right_pixel = 0
left_pixel = float('inf')

# Here we simply go through every x and y coordinate
# seeing which ones are black(text) and setting the topmost, bottommost, rightmost, and leftmost pixel.
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

# Now we use the topmost, bottommost, rightmost, and leftmost black pixel
# to crop the image so that this gets rid of unnecessary white background.
# We leave 5 pixels for the edge so it is easier to read.
cv2.rectangle(thresholded_img,(left_pixel, top_pixel),(right_pixel,bottom_pixel),(0,255,0),3)

# Now we crop the original clahe'd image.
cl1 = cl1[top_pixel-5:top_pixel+(bottom_pixel-top_pixel)+5, left_pixel-5:left_pixel+(right_pixel-left_pixel)+5]
# Dilate the image to blur everything and mostly get rid of the text
dilated_img = cv2.dilate(cl1, np.ones((7,7), np.uint8)) 
# Median blur the image to further get rid of the text.
background_blurred_img = cv2.medianBlur(dilated_img, 21)
diff_img = 255 - cv2.absdiff(cl1, background_blurred_img)
normalized_img = diff_img.copy()
cv2.normalize(diff_img, normalized_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

# Finally show the image
cv2.imshow('Output', normalized_img)

# Enter a loop so the program doesn't exit instantly and the user can see the image.
while True:
	# If q is pressed then exit out of our program.
    if cv2.waitKey(33) == ord('q'):
        cv2.destroyAllWindows()
        exit()