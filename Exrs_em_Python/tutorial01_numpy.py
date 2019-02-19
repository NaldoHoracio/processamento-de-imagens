# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 08:57:27 2019

@author: horacio
"""

import numpy as np

a = np.arange(15).reshape(3,5) # Create matrix 3x5 for start 1 to 15
a.shape # Dimentions of array
print("Dimenções da matriz:", a.ndim, "\n")
print("Tipo de dado:", a.dtype.name, "\n")
print("Tamanho do array:", a.itemsize, "\n")