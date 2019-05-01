from dip import *

input = cv.imread('img/horse.png',0)
size = np.size(input)
skel = np.zeros((input.shape[0], input.shape[1]),np.uint8)

ret,img = cv.threshold(input,127,255,0)
element = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
done = False

while(not done):
    eroded = cv.erode(img,element)
    temp = cv.dilate(eroded,element)
    temp = cv.subtract(img,temp)
    skel = cv.bitwise_or(skel,temp)
    img = eroded.copy()

    zeros = size - cv.countNonZero(img)
    if zeros==size:
        done = True

cv.imshow('Original', input)
cv.imshow("skel",skel)
cv.waitKey(0)
cv.destroyAllWindows()
