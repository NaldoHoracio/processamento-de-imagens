# -- coding: utf-8 --

# %%===========================================================================
# Import modules
# =============================================================================
#from math import log, e
from dip import *

#%% Q01 - Segmentar as rosas da imagem 'flowers.jpg'.
# img = cv.imread('imagens/flowers.jpg', cv.IMREAD_COLOR)
# img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# hsv_min = (0, 210,0)
# hsv_max = (15, 255, 255)
# mask = cv.inRange(img2, hsv_min, hsv_max)
# hsv_mask = mask > 0
# img_new = np.zeros_like(img, img.dtype)
# img_new[hsv_mask] = img[hsv_mask]
#
# while 0xFF & cv.waitKey(1) != ord('q'):
#     cv.imshow('Original', img)
#     cv.imshow('Segmentation', img_new)
# cv.destroyAllWindows()

#%% Q02 - Carregar a imagem 'baboon.png' e apresentar as 6 images
# corespondentes às camadas R (vermelha), G (verde), B (azul),
# C (ciano), M (magenta) e Y (amarelo).
# img = cv.imread('imagens/baboon.png', cv.IMREAD_COLOR)
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# r = np.array([1, 0, 0], dtype = np.float64)
# g = np.array([0, 1, 0], dtype = np.float64)
# b = np.array([0, 0, 1], dtype = np.float64)
# c = np.array([0, 1, 1], dtype = np.float64)
# m = np.array([1, 0, 1], dtype = np.float64)
# y = np.array([1, 1, 0], dtype = np.float64)
#
# plt.subplot('241')
# plt.title('RGB')
# plt.imshow(img)
# plt.subplot('242')
# plt.title('R')
# plt.imshow(r * img)
# plt.subplot('243')
# plt.title('G')
# plt.imshow(g * img)
# plt.subplot('244')
# plt.title('B')
# plt.imshow(b * img)
# plt.subplot('245')
# plt.title('C')
# plt.imshow(c * img)
# plt.subplot('246')
# plt.title('M')
# plt.imshow(m * img)
# plt.subplot('247')
# plt.title('Y')
# plt.imshow(y * img)
# plt.show()

#%% Q03 - A imagem 'abc.png' (também disponível no formato '.npy') corresponde
# à uma transformação do espaço de cores BGR ao espaço de cores ABC.
# Encontre a matriz rgb2abc 3x3 que efetua esta transformação.
# img3 = cv.imread('imagens/abc.png', cv.IMREAD_COLOR)
# abc = img3[0][0]
# bgr = img[0][0]
# print(abc, bgr)
# #transform = bgr.dot(np.linalg.pinv(abc))
# #print(transform)
#
# #img4 = np.dot(np.linalg.inv(transform), img3)
# while 0xFF & cv.waitKey(1) != ord('q'):
#     cv.imshow('ABC', img3)
# #    cv.imshow('Transformed', img4)
# cv.destroyAllWindows()
#
#%% Q04 - Deseja-se reconhecer duas poses de mão;
# (1) 'Aberta', e; (2) 'Fechada'.
# Aplique algum algoritmo morfológico às imagens 'hand1.jpg' e 'hand2.jpg'
# visando a identificação de cada uma dessas duas condições.

#%% Q05 - Desenvolva um algoritmo que visa obter o número de grãos de feijão
# presentes na imagem 'beans.png'
img = cv.imread('imagens/beans.png', cv.IMREAD_GRAYSCALE)
img_c = ~img
kernel = np.ones((3,3), dtype = np.uint8)
hitmiss = cv.morphologyEx(img, cv.MORPH_DILATE, kernel)
while 0xFF & cv.waitKey(1) != ord('q'):
    cv.imshow('Hit Miss', hitmiss)
cv.destroyAllWindows()
