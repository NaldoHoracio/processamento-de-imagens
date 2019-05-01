from dip import *

height = 512
width = 512
xc = height/2
yc = width/2
radius = 10
cv.namedWindow('Low Pass', cv.WINDOW_KEEPRATIO)
cv.createTrackbar('radius', 'Low Pass', radius, height/2, doNothing)
while 0xFF & cv.waitKey(1) != ord('q'):
    radius = cv.getTrackbarPos('radius', 'Low Pass')
    disk = createWhiteDisk(height, width, xc, yc, radius)
    cv.imshow('Low Pass', disk)
cv.destroyAllWindows()
