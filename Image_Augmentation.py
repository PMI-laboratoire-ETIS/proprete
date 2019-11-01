# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 07:11:14 2019

@author: Victor du Crest
"""

#%% imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras.preprocessing.image as IDG 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import os
from PIL import Image


#%% Load & crop
d = [p for p in os.listdir("images_intestins/") if "expert" in p]
d.sort(key=lambda x:int(x[6:-4])) # Retrier : 1, 10, 100, 2, 20, ..., 99 -> 1,2,3, ..., 600
# Prépare les images pour la lecture dans le réseau de neurones (resize à 150x150 car trop de mémoire requise sur mon ordi sinon)
x = np.asarray([np.array(Image.open("images_intestins/"+im).crop((32,32,544,544))) for im in d]) #.resize((150,150))
y = np.asarray(pd.read_csv("terrain.csv", header=None))[:,1]
        
#%% Tirage aléatoire de 500 images train et 100 images test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1/6, shuffle=False)

#%% Sauvegarde les images dans les bons fichiers : 'propre' & 'sale'
# Classification dans les fichiers 'train' et 'test' des images 'propres' et 'sales'
# TRAIN
for i in range(x_train.shape[0]):
    if y_train[i] == 0:
        img = Image.fromarray(x_train[i,:,:,:], 'RGB')
        img.save('.\\BDD\\train\\sale\\exp' + str(i+1) + '.png') 
    else:
        img = Image.fromarray(x_train[i,:,:,:], 'RGB')
        img.save('.\\BDD\\train\\propre\\exp' + str(i+1) + '.png')

# TEST
for i in range(x_test.shape[0]):
    if y_test[i] == 0:
        img = Image.fromarray(x_test[i,:,:,:], 'RGB')
        img.save('.\\BDD\\test\\sale\\exp' + str(i+1) + '.png')
    else:
        img = Image.fromarray(x_test[i,:,:,:], 'RGB')
        img.save('.\\BDD\\test\\propre\\exp' + str(i+1) + '.png')
        
#%% Création d'un modèle de génération aléatoire
# stockage des images dans le répertoire sous le nom 'propre_XXXX.png' ou 'sale_XXXX.png'
datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest') #Modèle du générateur automatique

sale = [p for p in os.listdir(".\\BDD\\train\\sale\\") if "exp" in p] 
propre = [p for p in os.listdir(".\\BDD\\train\\propre\\") if "exp" in p]

num_gen = 2 # Nombre d'image que l'on va générer par image

# Dans le répertoire sale on va récupérer les images et leur applique le générateur aléatoire
for nom in range(len(sale)):
    img = load_img('.\\BDD\\train\\sale\\' + sale[nom]) 
    x = img_to_array(img)  
    x = x.reshape((1,) + x.shape)  
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='.\\BDD\\train\\images_intestins_augmentées', 
                              save_prefix='sale', 
                              save_format='png'):
        i += 1
        if i > num_gen:
            break  

# Dans le répertoire propre on va récupérer les images et leur applique le générateur aléatoire
for nom in range(len(propre)):
    img = load_img('.\\BDD\\train\\propre\\' + propre[nom])  
    x = img_to_array(img)  
    x = x.reshape((1,) + x.shape)  
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='.\\BDD\\train\\images_intestins_augmentées', 
                              save_prefix='propre', 
                              save_format='png'):
        i += 1
        if i > num_gen:
            break  
