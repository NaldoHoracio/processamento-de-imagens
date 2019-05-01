from dip import *

img = cv.imread('img/horse.png', cv.IMREAD_GRAYSCALE)

#convert color image to binary
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert to grayscale
# blur = cv.blur(gray, (3, 3)) # blur the image
# use blur on img if colored
ret, thresh = cv.threshold(img, 127, 255, 0)
im2, contours, hierarchy = cv.findContours(thresh, 1, 2)
hull = cv.convexHull(im2)

while 0xFF & cv.waitKey(1) != ord('q'):
    cv.imshow('Original', img)
    cv.imshow('Convex Hull', hull)
cv.destroyAllWindows()
