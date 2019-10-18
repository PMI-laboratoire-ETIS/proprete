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
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from sklearn.model_selection import train_test_split

import os
from PIL import Image

batch_size = 20
epochs = 100

#%% Load & crop
d = [p for p in os.listdir("images_intestins/") if "expert" in p]
d.sort(key=lambda x:int(x[6:-4]))
x = np.asarray([np.array(Image.open("images_intestins/"+im).crop((32,32,544,544)).resize((150,150))) for im in d])
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
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer="adadelta",
              metrics=['accuracy'])

#%% Learn data
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

#%% Check accuracy of model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])