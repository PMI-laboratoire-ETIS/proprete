#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: pmi
"""
import os 
import matplotlib.pyplot as plt
import pandas as pd
import PIL as P
import numpy as np
import xlrd

classeur = xlrd.open_workbook('./Verite_Terrain.xls')
nom_des_feuilles = classeur.sheet_names()
feuille = classeur.sheet_by_name(nom_des_feuilles[0])

os.makedirs("image_triees/propre", exist_ok = True)
os.makedirs("image_triees/sale", exist_ok = True)

for i in range(1, 601): # lecture de toutes les images
    image_capture = P.Image.open('images_intestins/expert' + str(i) + '.png') 
    image_capture = image_capture.crop((32,32,image_capture.width-32,image_capture.height-32)) # Crop
    image_capture.paste(0, (0,24,11,39)) # Remove white dash
    image_capture = image_capture.resize((64,64))
#image_prediction = np.asarray(image_capture)/255
    
    if (feuille.cell_value(i-1, 1) == 0.0) : 
        image_capture.save('./image_triees/sale/image'+ str(i) + '.jpg')
    if (feuille.cell_value(i-1, 1) == 1.0) : 
        image_capture.save('./image_triees/propre/image'+ str(i) + '.jpg')
        
    