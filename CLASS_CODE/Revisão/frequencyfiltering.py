from dip import *

height = width = 512
freq = 500
theta = 10

img = np.float64(cv.imread('img/lena.png', cv.IMREAD_GRAYSCALE))/255
img2 = createCosineImage2(height, width, freq/1e3, theta)
noised = img + img2

xc = height/2
yc = width/2
radius1 = 10
radius2 = 250
dimensions = noised.shape

planes = [np.zeros(dimensions, dtype=np.float64), np.zeros(dimensions, dtype=np.float64)]
planes[0] = noised

img3 = cv.merge(planes)
img_freq = cv.dft(img3)
fft = np.fft.fftshift(img_freq)
planes = cv.split(fft)
magnitude = np.zeros(dimensions, dtype=np.float64)
phase = np.zeros(dimensions, dtype=np.float64)
cv.cartToPolar(planes[0], planes[1], magnitude, phase)

spectrum = magnitude + 1
spectrum = np.log(spectrum)
cv.normalize(spectrum, spectrum, 0, 1, cv.NORM_MINMAX)
cv.namedWindow('Band Reject', cv.WINDOW_KEEPRATIO)
cv.createTrackbar('radius 1', 'Band Reject', radius1, height/2, doNothing)
cv.createTrackbar('radius 2', 'Band Reject', radius2, height/2, doNothing)
while 0xFF & cv.waitKey(1) != ord('q'):
    cv.imshow('Noised Image', noised)
    cv.imshow('Spectrum', spectrum)
    radius1 = cv.getTrackbarPos('radius 1', 'Band Reject')
    radius2 = cv.getTrackbarPos('radius 2', 'Band Reject')
    disk = createBlackRing2(height, width, xc, yc, radius1, radius2)
    cv.imshow('Band Reject', disk)
    filtered = disk * spectrum
    filtered = np.exp(filtered)
    filtered -= 1
    cv.polarToCart(magnitude, phase, planes[0], planes[1])
    ifft = cv.merge(planes)
    ifft = np.fft.ifftshift(fft)
    new_img = cv.idft(ifft)
    planes = cv.split(new_img)
    ret = planes[0]
    cv.normalize(ret, ret, 0, 1, cv.NORM_MINMAX)
    cv.imshow('filtered Image', filtered)
    cv.imshow('Result', ret)
cv.destroyAllWindows()
