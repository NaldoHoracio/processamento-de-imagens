# Importando as bibliotecas necessarias para processar as imagens e criar as
# camadas da rede neural
from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Dropout
#from keras.layers.convolution import Conv2D
from keras.layers.convolutional import Conv2D
#from keras.layers.convolution import MaxPooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
from keras.optimizers import SGD
from keras import backend

import matplotlib.pyplot as plt
import numpy as np
import glob as gl
import cv2 as cv
import os

RGB = 3
GRAY = 1

path_db = "/home/horacio/Documentos/4-PROCESSAMENTO_DE_IMAGENS-8-/KeratoDetect/DataBase/Eyes"

class KeratoNet(object):
    @staticmethod
    # HEIGHT = ALTURA DA IMG; WIDTH = COMPRIMENTO DA IMG; CHANNELS = RGB;
    # CLASSES = NUM DE CLASSIFICAÇÃO - NORMAL OU KT
    def buildNet(width, height, channels, classes):
        
        inputShape = (width, height, channels)

        # Declaramos a nossa rede seguindo um modelo sequencial.
        model = Sequential()

        # Em seguida, adicionamos as camadas da rede neural, seguindo a sequencia:
        # Convolucao => Normalizacao => Ativacao => Pooling
        # Fazendo esta sequencia 3 vezes, aumentando o numero de filtros utilizados
        # de 16, para 32, e em seguida 64.
        # Utilizaremos um kernel de tamanho 3x3.
#        model.add(Conv2D(6, (5, 5), padding="same",
#                         input_shape=inputShape))
#        model.add(Activation("relu"))
#        
        model.add(Conv2D(filters = 16, kernel_size = (3,3), padding="same", 
                            input_shape = inputShape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(filters = 32, kernel_size = (3,3), padding="same", 
                             input_shape = inputShape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(filters = 64, kernel_size = (3,3), padding="same", 
                             input_shape = inputShape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))

        # Adicionamos aqui a camada totalmente conectada da rede, seguida da camada
        # softmax, que fará a classificação.
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model