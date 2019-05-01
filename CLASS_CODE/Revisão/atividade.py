from dip import *

img = cv.imread('img/chips.png', cv.IMREAD_COLOR)
img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)
hsv = cv.split(img2)

# CANAIS H, S, V
plt.figure(1)
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

# HISTOGRAMAS
plt.figure(2)
plt.subplot('131')
plt.hist(hsv[0].ravel(), 256, (0, 255))
plt.title('Hue Histogram')
plt.subplot('132')
plt.hist(hsv[1].ravel(), 256, (0, 255))
plt.title('Saturation Histogram')
plt.subplot('133')
plt.hist(hsv[2].ravel(), 256, (0, 255))
plt.title('Value Histogram')

#%%
img = cv.imread('img/chips.png', cv.IMREAD_COLOR)
img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)
hsv = cv.split(img2)

# SEGMENTANDO CHIPS DO BACKGROUND
plt.figure(3)
hsv_min = (0, 160, 0)
hsv_max = (255, 255, 255)
mask = cv.inRange(img2, hsv_min, hsv_max)
mask_chips = mask > 0
chips = np.zeros_like(img, img.dtype)
chips[mask_chips] = img[mask_chips]
plt.subplot('231')
plt.imshow(cv.cvtColor(chips, cv.COLOR_BGR2RGB))
plt.title('Chips')

# SEGMENTANDO CHIPS LARANJA
hsv_min = (0, 160, 224)
hsv_max = (20, 255, 255)
mask = cv.inRange(img2, hsv_min, hsv_max)
mask_orange = mask > 0
orange = np.zeros_like(img, img.dtype)
orange[mask_orange] = img[mask_orange]
plt.subplot('232')
plt.imshow(cv.cvtColor(orange, cv.COLOR_BGR2RGB))
plt.title('Chips Laranja')

# SEGMENTANDO CHIPS AMARELO
hsv_min = (25, 160, 0)
hsv_max = (60, 255, 255)
mask = cv.inRange(img2, hsv_min, hsv_max)
mask_orange = mask > 0
orange = np.zeros_like(img, img.dtype)
orange[mask_orange] = img[mask_orange]
plt.subplot('233')
plt.imshow(cv.cvtColor(orange, cv.COLOR_BGR2RGB))
plt.title('Chips Laranja')

# SEGMENTANDO CHIPS VERDE
hsv_min = (60, 160, 0)
hsv_max = (100, 255, 255)
mask = cv.inRange(img2, hsv_min, hsv_max)
mask_green = mask > 0
green = np.zeros_like(img, img.dtype)
green[mask_green] = img[mask_green]
plt.subplot('234')
plt.imshow(cv.cvtColor(green, cv.COLOR_BGR2RGB))
plt.title('Chips Verde')

# SEGMENTANDO CHIPS AZUL
hsv_min = (100, 160, 0)
hsv_max = (140, 255, 255)
mask = cv.inRange(img2, hsv_min, hsv_max)
mask_blue = mask > 0
blue = np.zeros_like(img, img.dtype)
blue[mask_blue] = img[mask_blue]
plt.subplot('235')
plt.imshow(cv.cvtColor(blue, cv.COLOR_BGR2RGB))
plt.title('Chips Azul')

# SEGMENTANDO CHIPS VERMELHO
hsv_min = (150, 160, 0)
hsv_max = (180, 255, 255)
mask = cv.inRange(img2, hsv_min, hsv_max)
mask_red = mask > 0
red = np.zeros_like(img, img.dtype)
red[mask_red] = img[mask_red]
plt.subplot('236')
plt.imshow(cv.cvtColor(red, cv.COLOR_BGR2RGB))
plt.title('Chips Vermelho')

plt.show()
