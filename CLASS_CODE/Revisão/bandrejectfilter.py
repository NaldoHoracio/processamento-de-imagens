from dip import *

height = 512
width = 512
xc = height/2
yc = width/2
radius1 = 10
radius2 = 20
cv.namedWindow('Band Reject', cv.WINDOW_KEEPRATIO)
cv.createTrackbar('radius 1', 'Band Reject', radius1, int (height/2), doNothing)
cv.createTrackbar('radius 2', 'Band Reject', radius2, int (height/2), doNothing)
while 0xFF & cv.waitKey(1) != ord('q'):
    radius1 = cv.getTrackbarPos('radius 1', 'Band Reject')
    radius2 = cv.getTrackbarPos('radius 2', 'Band Reject')
    disk = createBlackRing2(height, width, xc, yc, radius1, radius2)
    cv.imshow('Band Reject', disk)
cv.destroyAllWindows()
