# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:59:49 2019

@author: horacio
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

idx = 31
calculator = cv2.imread(os.path.join('/home/horacio/Imagens/PDI', 'calculator.tif'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((idx,idx), np.uint8)
remove = cv2.morphologyEx(calculator, cv2.MORPH_OPEN, kernel)

plt.subplot('131'), plt.imshow(calculator, cmap = 'gray'), plt.title('Reflections wider than characters')
plt.subplot('132'), plt.imshow(remove, cmap = 'gray'), plt.title('Opening by Reconstruction (OBR)')

plt.show()