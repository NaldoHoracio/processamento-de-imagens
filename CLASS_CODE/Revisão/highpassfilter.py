from dip import *

height = 512
width = 512
xc = height/2
yc = width/2
radius = 10
cv.namedWindow('High Pass', cv.WINDOW_KEEPRATIO)
cv.createTrackbar('radius', 'High Pass', radius, height/2, doNothing)
while 0xFF & cv.waitKey(1) != ord('q'):
    radius = cv.getTrackbarPos('radius', 'High Pass')
    disk = createBlackDisk2(height, width, xc, yc, radius)
    cv.imshow('High Pass', disk)
cv.destroyAllWindows()
