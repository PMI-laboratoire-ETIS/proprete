#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:48:08 2019

@author: lea
"""

#%% imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
from time import process_time  
from skimage import color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
import matplotlib.image as mpimg

#%%
def hough_fonction(image):
    image = color.rgb2gray(image)
    edges = canny(image, sigma=3, mask = image>1e-2)
    plt.imshow(edges)
    hough_radii = np.arange(5, 30, 10)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_xdistance=20, min_ydistance=20)
    return np.size(radii)


#%%


t1_start = process_time()  

d = [p for p in os.listdir("/Users/lea/Documents/cours/A5/PMI/images_intestins") if "expert" in p]
d.sort(key=lambda x:int(x[6:-4])) # Retrier : 1, 10, 100, 2, 20, ..., 99 -> 1,2,3, ..., 600
x = np.asarray([np.array(Image.open("/Users/lea/Documents/cours/A5/PMI/images_intestins/"+im).crop((32,32,544,544))) for im in d]) 
y = np.asarray(pd.read_excel("/Users/lea/Documents/cours/A5/PMI/Verite_Terrain.xls", header=None))[:,1]

input_shape = x.shape[1:]

RG_ratio= np.zeros(len(x))
LUM_ratio=np.zeros(len(x))
critere= np.zeros((len(x),3))
bulle_ratio = np.zeros(len(x))

RG_ratio = x[:,:,:,0].sum(axis=(1,2))/x[:,:,:,1].sum(axis=(1,2))   
LUM_ratio = (x[:,:,:,0].sum(axis=(1,2))+ x[:,:,:,1].sum(axis=(1,2)) + x[:,:,:,2].sum(axis=(1,2)))/ (input_shape[0]* input_shape[1]) 

critere[:, 0]= RG_ratio>1.6
critere[:,2]= LUM_ratio > (255/2)

for i in range(0, len(x)): 
    bulle_ratio[i]= hough_fonction(x[i])
    
    if bulle_ratio[i] > 150: 
        critere[(i, 1)]=0
    else : 
        critere[(i, 1)]=1
        
t1_stop = process_time()    
   
print("Elapsed time:", t1_stop, t1_start)  
   
print("Elapsed time during the whole program in seconds:",  t1_stop-t1_start)  
 

#%%
resultat = critere[:,0]*2 + critere[:,1]*2+ critere[:,2]
resultat_final = (resultat>2).astype(np.int8)


#from sklearn.metrics import precision_score, recall_score
#precision_score(y, resultat_final)
#recall_score(y, resultat_final)
#%%

path = "/Users/lea/Documents/cours/A5/PMI"
fichier = open("/Users/lea/Documents/cours/A5/PMI/Nomdufichier.txt", "w")

for i in range(0, len(resultat_final)):
    a = str(resultat_final[i])
    fichier.write(a+'\n')

fichier.close()


#%%

#resultat = np.sqrt(RG_ratio * LUM_ratio * 1/(bulle_ratio+1))
#    
#fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
#image1 = color.gray2rgb(image)
#
#for center_y, center_x, radius in zip(cy, cx, radii):
#    circy, circx = circle_perimeter(center_y, center_x, radius,
#                                    shape=image1.shape)
#    image1[circy, circx] = (220, 5, 20)
#
#ax.imshow(image1, cmap=plt.cm.gray)
#plt.show()

