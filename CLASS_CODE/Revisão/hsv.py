from dip import *

img = cv.imread('img/chips.png', cv.IMREAD_COLOR)
img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)
hsv = cv.split(img2)

plt.subplot('221')
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original')
plt.subplot('222')
plt.imshow(hsv[0], 'gray')
plt.title('Hue')
plt.subplot('223')
plt.imshow(hsv[1], 'gray')
plt.title('Saturation')
plt.subplot('224')
plt.imshow(hsv[2], 'gray')
plt.title('Value')
plt.show()

h_min = 0
s_min = 0
v_min = 0
h_max = 255
s_max = 255
v_max = 255
cv.namedWindow('Segmentation', cv.WINDOW_KEEPRATIO)
cv.createTrackbar( 'H min','Segmentation', h_min, 255, doNothing)
cv.createTrackbar('H max', 'Segmentation', h_max, 255, doNothing)
cv.createTrackbar('S min', 'Segmentation', s_min, 255, doNothing)
cv.createTrackbar('S max', 'Segmentation', s_max, 255, doNothing)
cv.createTrackbar('V min', 'Segmentation', v_min, 255, doNothing)
cv.createTrackbar('V max', 'Segmentation', v_max, 255, doNothing)

while 0xFF & cv.waitKey(1) != ord('q'):
    h_min = cv.getTrackbarPos('H min', 'Segmentation')
    s_min = cv.getTrackbarPos('S min', 'Segmentation')
    v_min = cv.getTrackbarPos('V min', 'Segmentation')
    h_max = cv.getTrackbarPos('H max', 'Segmentation')
    s_max = cv.getTrackbarPos('S max', 'Segmentation')
    v_max = cv.getTrackbarPos('V max', 'Segmentation')
    hsv_min = (h_min, s_min, v_min)
    hsv_max = (h_max, s_max, v_max)
    mask = cv.inRange(img2, hsv_min, hsv_max)
    hsv_mask = mask > 0
    img_new = np.zeros_like(img, img.dtype)
    img_new[hsv_mask] = img[hsv_mask]
    cv.imshow('Segmentation', img_new)
cv.destroyAllWindows()
