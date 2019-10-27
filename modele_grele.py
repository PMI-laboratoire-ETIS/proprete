# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:20:21 2019

@author: julien
"""

#%% imports
import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_curve

import matplotlib.pyplot as plt

import os
from PIL import Image

batch_size = 32
epochs = 100
imdir = "images_intestins"

#%% Load & crop
d = [p for p in os.listdir(imdir) if "png" in p]
d.sort(key=lambda x:int(x[6:-4])) # Retrier : 1, 10, 100, 2, 20, ..., 99 -> 1,2,3, ..., 600
# Prépare les images pour la lecture dans le réseau de neurones (resize à 150x150 car trop de mémoire requise sur mon ordi sinon)
#x = np.asarray([np.array(Image.open(os.path.join(imdir, im)).crop((32,32,544,544)).resize((150,150))) for im in d]) 
x = np.asarray([np.array(Image.open(os.path.join(imdir, im)).resize((150,150))) for im in d]) 
y = np.asarray(pd.read_excel("Verite_Terrain.xls", header=None))[:,1]

input_shape = x.shape[1:]

#%% Split train and test data, normalize
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/6)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train /= 255
x_test /= 255
#%% Network creation

model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(196, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.2)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.2)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer="adam",
              metrics=['accuracy'])

#%% Learn data
history = model.fit(x_train, y_train,
                    
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

#%% Plot history of learning process
figm, axs = plt.subplots(1, 2, clear=True, num="Metrics")
axs[0].semilogy(history.history["loss"], label="loss")
axs[0].semilogy(history.history["val_loss"], label="val_loss")
axs[0].legend()
axs[0].set_xlim(0)
axs[0].set_title("Loss evolution (semilogy)")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss (binary cross-entropy)")
axs[1].plot(history.history["acc"], label="acc")
axs[1].plot(history.history["val_acc"], label="val_acc")
axs[1].legend()
axs[1].set_xlim(0)
axs[1].set_ylim(0,1)
axs[1].set_title("Accuracy evolution")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy")

#%% Check accuracy of model
y_pred = model.predict(x_test)
y_pred_classes = model.predict_classes(x_test)

sensitivity = recall_score(y_test, y_pred_classes)
specificity = recall_score(1-y_test, 1-y_pred_classes)

print("Sensitivity: {:.2%}".format(sensitivity))
print("Specificity: {:.2%}".format(specificity))

fpr, tpr = roc_curve(y_test, y_pred)[:-1]

figroc, axroc = plt.subplots(1, 1, clear=True, num="Courbe ROC")
axroc.plot(fpr, tpr, label="Covnet")
axroc.plot([0,1],[0,1],'r--')
axroc.set_xlim(0,1)
axroc.set_ylim(0,1)
axroc.scatter(1 - specificity, sensitivity, marker="d", label="Operating point")
axroc.legend()