# propreté

Séparations d'images propres et sales issues d'un examen de l'intestin grêle par une pilule Pillcam© à l'aide d'un réseau de neurones convolutif reposant sur Keras.


Structure du dossier attendue pour faire fonctionner le programme actuel :

```
proprete
│
├── modele_grele.py
├── Verite_Terrain.xls
└── images_intestins
    ├── expert1.png
    ├── expert2.png
    │   ...
    └── expert600.png
```

En entraînant le réseau sur 500 images et en le testant sur les 100 restantes, nous avons obtenu une précision de 99.4% sur l'ensemble d'entraînement, et de 93% sur l'ensemble de test.

Architecture du réseau :
```
________________________________________________________________________________________________
Layer (type)                               Output Shape                          Param #        
================================================================================================
conv2d_1 (Conv2D)                          (None, 148, 148, 32)                  896            
________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)             (None, 74, 74, 32)                    0              
________________________________________________________________________________________________
conv2d_2 (Conv2D)                          (None, 72, 72, 32)                    9248           
________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)             (None, 36, 36, 32)                    0              
________________________________________________________________________________________________
conv2d_3 (Conv2D)                          (None, 34, 34, 64)                    18496          
________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)             (None, 17, 17, 64)                    0              
________________________________________________________________________________________________
flatten_1 (Flatten)                        (None, 18496)                         0              
________________________________________________________________________________________________
dense_1 (Dense)                            (None, 128)                           2367616        
________________________________________________________________________________________________
dropout_1 (Dropout)                        (None, 128)                           0              
________________________________________________________________________________________________
dense_2 (Dense)                            (None, 1)                             129            
================================================================================================
Total params: 2,396,385
Trainable params: 2,396,385
Non-trainable params: 0
________________________________________________________________________________________________
```

Résultats actuels :
Sensibilité : 93.55%
Spécificité : 88.41%

Courbe ROC :
![Alt text](./graphs/Courbe_ROC.svg)
<img src="./graphs/Courbe_ROC.svg">