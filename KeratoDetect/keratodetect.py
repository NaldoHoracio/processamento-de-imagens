from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import fetch_mldata
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD

import matplotlib.pyplot as plt
from cnn import KeratoNet
import numpy as np
import glob as gl
import cv2 as cv
import os

# Carregar o banco de imagens ^ separar em dois grupos
# Ok!

#path_ne = "/home/horacio/Documentos/4-PROCESSAMENTO_DE_IMAGENS-8-/KeratoDetect/DataBase/Normal_Eyes"
#path_kt = "/home/horacio/Documentos/4-PROCESSAMENTO_DE_IMAGENS-8-/KeratoDetect/DataBase/Kt_Eyes"

path_db = "/home/horacio/Documentos/4-PROCESSAMENTO_DE_IMAGENS-8-/KeratoDetect/DataBase/Eyes"
## Separar em dois grupos: Treino e Teste
#
## Carregando imagens
#fold_eye = gl.glob(path_db)
#
#imgs_eyes_list = []
#
#for fold in fold_eye:
#    for f in gl.glob(fold + '/*.png'):
#        imgs_eyes_list.append(f)
#
#read_fold = []
#
#for img in imgs_eyes_list:
#    read_fold.append(cv.imread(img, cv.IMREAD_COLOR))

#print(read_fold)

kn_cnn = KeratoNet.buildNet(180,240,3,2)

## plotar loss e accuracy para os datasets 'train' e 'test'
#print("[INFO] treinando a rede neural...")
#model.compile(optimizer=SGD(0.01), loss="categorical_crossentropy",
#             metrics=["accuracy"])
#H = model.fit(trainX, trainY, batch_size=128, epochs=10, verbose=2,
#         validation_data=(testX, testY))
#
## avaliar a Rede Neural
#print("[INFO] avaliando a rede neural...")
#predictions = model.predict(testX, batch_size=128)
#print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))
#
## plotar loss e accuracy para os datasets 'train' e 'test'
#plt.style.use("ggplot")
#plt.figure()
#plt.plot(np.arange(0,100), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0,100), H.history["val_loss"], label="val_loss")
#plt.plot(np.arange(0,100), H.history["acc"], label="train_acc")
#plt.plot(np.arange(0,100), H.history["val_acc"], label="val_acc")
#plt.title("Training Loss and Accuracy")
#plt.xlabel("Epoch #")
#plt.ylabel("Loss/Accuracy")
#plt.legend()
#plt.show()


## Carregando imagens de olhos saudáveis
#fold_normal_eye = gl.glob(path_ne)
#
#imgs_normal_eyes_list = []
#
#for fold_ne in fold_normal_eye:
#    for f_ne in gl.glob(fold_ne + '/*.png'):
#        imgs_normal_eyes_list.append(f_ne)
#
#read_ne = []
#
#for img_ne in imgs_normal_eyes_list:
#    read_ne.append(cv.imread(img_ne, cv.IMREAD_COLOR))
#
##print(read_ne)
##print("\n")
#
## Carregando imagens de olhos com kt
#fold_kt_eye = gl.glob(path_kt)
#
#imgs_kt_eyes_list = []
#
#for fold_kt in fold_kt_eye:
#    for f_kt in gl.glob(fold_kt + '/*.jpg'):
#        imgs_kt_eyes_list.append(f_kt)
#
#read_kt = []
#
#for img_kt in imgs_kt_eyes_list:
#    read_kt.append(cv.imread(img_kt, cv.IMREAD_COLOR))

#print(read_kt)

# Instanciar a KeratoNet
#kn_cnn = KeratoNet.buildNet(180,240,3,2)
