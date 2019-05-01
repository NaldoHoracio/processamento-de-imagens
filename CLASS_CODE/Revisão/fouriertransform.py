from dip import *

img = np.float64(cv.imread('img/lena.png', cv.IMREAD_GRAYSCALE))/255
dimensions = img.shape
planes = [np.zeros(dimensions, dtype=np.float64), np.zeros(dimensions, dtype=np.float64)]
planes[0] = img
img2 = cv.merge(planes)
img_freq = cv.dft(img2)
fft = np.fft.fftshift(img_freq)
planes = cv.split(fft)
magnitude = np.zeros(dimensions, dtype=np.float64)
phase = np.zeros(dimensions, dtype=np.float64)
cv.cartToPolar(planes[0], planes[1], magnitude, phase)
magnitude += 1
magnitude = np.log(magnitude)
new_magnitude = np.zeros(dimensions, dtype=np.float64)
cv.normalize(magnitude, new_magnitude, 0, 1, cv.NORM_MINMAX)
magnitude = np.exp(magnitude)
magnitude -= 1
cv.polarToCart(magnitude, phase, planes[0], planes[1])
ifft = cv.merge(planes)
ifft = np.fft.ifftshift(ifft)
new_img = cv.idft(ifft)
planes = cv.split(new_img)
ret = planes[0]
cv.normalize(ret, ret, 0, 1, cv.NORM_MINMAX)
while 0xFF & cv.waitKey(1) != ord('q'):
    cv.imshow('Original', img)
    cv.imshow('Magnitude', new_magnitude)
    cv.imshow('Returned', ret)
cv.destroyAllWindows()
