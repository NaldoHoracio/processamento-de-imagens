from dip import *

freq = 1
theta = 0

img = scaleImage2_uchar(cv.imread('img/lena.png', cv.IMREAD_GRAYSCALE))

cv.namedWindow('Noise', cv.WINDOW_KEEPRATIO)
cv.createTrackbar('frequency', 'Noise', freq, 500, doNothing)
cv.createTrackbar('theta', 'Noise', theta, 360, doNothing)

while 0xFF & cv.waitKey(1) != ord('q'):
    freq = cv.getTrackbarPos('frequency', 'Noise') / 1e3
    theta = cv.getTrackbarPos('theta', 'Noise')
    noise =  scaleImage2_uchar(createCosineImage2(img.shape[0], img.shape[1], freq, theta))
    noised = img + noise
    cv.imshow('Original', img)
    cv.imshow('Noise', noise)
    cv.imshow('Result', noised)
cv.destroyAllWindows()
