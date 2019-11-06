# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:34:06 2019

@author: Victor du Crest
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

imdir = "images_intestins"

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

input_shape = (512, 512, 3) 

train_generator = datagen.flow_from_directory(
        imdir,
        target_size=input_shape[:2],
        batch_size=32,
        class_mode='binary',
        classes=['propre', 'sale'],
        subset='training')

validation_generator = datagen.flow_from_directory(
        imdir,
        target_size=input_shape[:2],
        batch_size=32,
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



model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)