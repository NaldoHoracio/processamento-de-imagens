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

import matplotlib.pyplot as plt
import numpy as np
import glob as gl
import cv2 as cv
import os

RGB = 3
GRAY = 1

path_db = "/home/horacio/Documentos/4-PROCESSAMENTO_DE_IMAGENS-8-/KeratoDetect/DataBase/Eyes"

class KeratoNet:
    # HEIGHT = ALTURA DA IMG; WIDTH = COMPRIMENTO DA IMG; CHANNELS = RGB;
    # CLASSES = NUM DE CLASSIFICAÇÃO - NORMAL OU KT
    def buildNet(height, width, channels, classes):
        # Carregando imagens
        fold_eye = gl.glob(path_db)

        imgs_eyes_list = []

        for fold in fold_eye:
            for f in gl.glob(fold + '/*.png'):
                imgs_eyes_list.append(f)

        read_fold = []
        for img in imgs_eyes_list:
            read_fold.append(cv.imread(img, cv.IMREAD_COLOR))
#        
#        # normalizar todos pixels, de forma que os valores estejam
#        # no intervalor [0, 1.0]
#        data = []
#        #data = read_fold("int")/255
#                     
#            
#        # dividir o dataset entre train (75%) e test (25%)
#        (trainX, testX, trainY, testY) = train_test_split(data, img_eyes_list.target)
# 
#        # converter labels de inteiros para vetores
#        lb = LabelBinarizer()
#        trainY = lb.fit_transform(trainY)
#        testY = lb.transform(testY)

        # Declaramos a nossa rede seguindo um modelo sequencial.
        model = Sequential()

        # Em seguida, adicionamos as camadas da rede neural, seguindo a sequencia:
        # Convolucao => Normalizacao => Ativacao => Pooling
        # Fazendo esta sequencia 3 vezes, aumentando o numero de filtros utilizados
        # de 16, para 32, e em seguida 64.
        # Utilizaremos um kernel de tamanho 3x3.
        model.add(Conv2D(filters = 16, kernel_size = (3,3), input_shape = (height, width, channels)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = (height, width, channels)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(filters = 64, kernel_size = (3,3), input_shape = (height, width, channels)))
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
        
#        print("[INFO] treinando a rede neural...")
#        model.compile(optimizer=SGD(0.01), loss="categorical_crossentropy", metrics=["accuracy"])
#        H = model.fit(trainX, trainY, batch_size=128, epochs=10, verbose=2, validation_data=(testX, testY))
#        
#        # avaliar a Rede Neural
#        print("[INFO] avaliando a rede neural...")
#        predictions = model.predict(testX, batch_size=128)
#        print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))
#        
#        # plotar loss e accuracy para os datasets 'train' e 'test'
#        plt.style.use("ggplot")
#        plt.figure()
#        plt.plot(np.arange(0,100), H.history["loss"], label="train_loss")
#        plt.plot(np.arange(0,100), H.history["val_loss"], label="val_loss")
#        plt.plot(np.arange(0,100), H.history["acc"], label="train_acc")
#        plt.plot(np.arange(0,100), H.history["val_acc"], label="val_acc")
#        plt.title("Training Loss and Accuracy")
#        plt.xlabel("Epoch #")
#        plt.ylabel("Loss/Accuracy")
#        plt.legend()
#        plt.show()

        return model
