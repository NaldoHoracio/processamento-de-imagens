from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import fetch_mldata
from keras.utils import to_categorical
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras import backend

import matplotlib.pyplot as plt
from cnn import KeratoNet
import numpy as np
import glob as gl
import cv2 as cv
import os

path_db = "/home/horacio/Documentos/4-PROCESSAMENTO_DE_IMAGENS-8-/KeratoDetect/DataBase/Eyes"


# CARREGAR O BANCO DE IMAGENS E NORMALIZANDO

fold_eye = gl.glob(path_db)

imgs_eyes_list = []

for fold in fold_eye:
    for f in gl.glob(fold + '/*.png'):
        imgs_eyes_list.append(f)

read_fold = [] # Arquivo com dados originais das imagens em RGB
read_fold_norm = [] # Arquivo com dados normalizados das imagens em RBG
read_fold_norm_s = []

for img in imgs_eyes_list:
    read_fold.append(cv.imread(img, cv.IMREAD_COLOR))
    read_fold_norm.append(cv.imread(img, cv.IMREAD_COLOR))

labels = read_fold

for i in range(0,40):
    read_fold_norm[i] = read_fold_norm[i].astype('float32')
    read_fold_norm[i] /= 255.0

read_fold_norm_s = np.asarray(read_fold_norm)

# converter as imagens de 1D para o formato (28x28x1)
if backend.image_data_format() == "channels_last":
    read_fold_norm_s = read_fold_norm_s.reshape(
                                (read_fold_norm_s.shape[0], 180, 240, 3))
else:
    read_fold_norm_s = read_fold_norm_s.reshape(
                                (read_fold_norm_s.shape[0], 3, 180, 240))

# SEPARAR EM DOIS GRUPOS: TREINO E TESTE

# dividir o dataset entre train (75%) e test (25%)
(trainX_kt, testX_kt, trainY_kt, testY_kt) = train_test_split(read_fold_norm_s,
                                                read_fold_norm_s)
 
# Transformar labels em vetores binarios
trainY_kt = to_categorical(trainY_kt, 2)
testY_kt = to_categorical(testY_kt, 2)

# INSTANCIAR A KeratoNet
# inicializar e otimizar modelo
print("[INFO] inicializando e otimizando a CNN...")
model = KeratoNet.buildNet(180, 240, 3, 2)
model.compile(optimizer=SGD(0.01), loss="binary_crossentropy",
              metrics=["accuracy"])
 
# treinar a CNN
print("[INFO] treinando a CNN...")
H = model.fit(trainX_kt, trainY_kt, batch_size=64, epochs=5, verbose=2,
          validation_data=(testX_kt, testY_kt))

# avaliar a CNN
print("[INFO] avaliando a CNN...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(label) for label in range(2)]))

# PLOTAR loss E accuracy PARA OS DATABASE 'train' E 'TEST'
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,5), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,5), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,5), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,5), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('cnn.png', bbox_inches='tight')
        
