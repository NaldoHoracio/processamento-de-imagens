from dip import *

height = 512
width = 512
xc = int (height/2)
yc = int (width/2)
radius1 = 10
radius2 = 20
cv.namedWindow('Band Pass', cv.WINDOW_KEEPRATIO)
cv.createTrackbar('radius 1', 'Band Pass', radius1, int(height/2), doNothing)
cv.createTrackbar('radius 2', 'Band Pass', radius2, int(height/2), doNothing)
while 0xFF & cv.waitKey(1) != ord('q'):
    radius1 = cv.getTrackbarPos('radius 1', 'Band Pass')
    radius2 = cv.getTrackbarPos('radius 2', 'Band Pass')
    disk = createWhiteRing2(height, width, xc, yc, radius1, radius2)
    cv.imshow('Band Pass', disk)
cv.destroyAllWindows()
