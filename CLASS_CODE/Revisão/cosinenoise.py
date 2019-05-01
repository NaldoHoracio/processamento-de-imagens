from dip import *

height = width = 512
freq = 50
theta = 30
img = np.float64(cv.imread('img/lena.png', cv.IMREAD_GRAYSCALE))/255
cv.namedWindow('Cosine', cv.WINDOW_KEEPRATIO)
cv.createTrackbar('frequency', 'Cosine', freq, 10000, doNothing)
cv.createTrackbar('theta', 'Cosine', theta, 360, doNothing)
while 0xFF & cv.waitKey(1) != ord('q'):
    freq = cv.getTrackbarPos('frequency', 'Cosine')
    theta = cv.getTrackbarPos('theta', 'Cosine')
    img2 = createCosineImage2(height, width, freq, theta)
    noised = img * img2
    cv.imshow('Cosine', noised)
cv.destroyAllWindows()
