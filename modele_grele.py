# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:34:06 2019

@author: groupe pmi
"""


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from sklearn.metrics import recall_score, roc_curve

import matplotlib.pyplot as plt

imdir = "images_intestins"
batch_size = 32
input_shape = (512, 512, 3) 

def bruit(image, λ):
    img = image + np.random.normal(size=image.shape) * image * λ
    img[img < 0] = 0 ; img[img > 255] = 255
    return img

datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant',
        cval = 0,
        validation_split=1/6,
        preprocessing_function=lambda img : bruit(img, 0.1)) #Modèle du générateur automatique


train_generator = datagen.flow_from_directory(
        imdir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='binary',
        classes=['propre', 'sale'],
        subset='training')

validation_generator = datagen.flow_from_directory(
        imdir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        classes=['propre', 'sale'],
        class_mode='binary',
        subset='validation')

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


history = model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)

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
axroc.plot(fpr, tpr, label="CNN")
axroc.plot([0,1],[0,1],'r--')
axroc.set_xlim(0,1)
axroc.set_ylim(0,1)
axroc.scatter(1 - specificity, sensitivity, marker="d", label="Point d'opération")
axroc.legend()