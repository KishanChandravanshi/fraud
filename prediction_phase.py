# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:09:42 2019

@author: Kishan Kumar
"""

import pandas as pd
import numpy as np

import keras
from keras_retinanet import models
from keras_retinanet.models import load_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import keras_retinanet
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import cv2
import tensorflow as tf
# keras libraries
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, Activation, BatchNormalization, Dropout
from keras.layers.merge import add
from keras.models import Model, load_model
from keras import backend as k
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import img_to_array

model = keras_retinanet.models.backbone('resnet50').retinanet(num_classes=1)
labels_to_names = {0: 'center'}
model.compile(
    loss={
        'regression'    : keras_retinanet.losses.smooth_l1(),
        'classification': keras_retinanet.losses.focal()
    },
    optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
)


model = models.load_model("model.h5", backbone_name='resnet50')
#print(model.summary())

foldername = "images"
predictionFolder = "Pred"
import glob
# access all the images present in the folder
imagesPath = glob.glob(foldername + "\\*.jpg")

print("{} images were found".format(len(imagesPath)))

path="F:/keras-retinanet/"

centers = []

for full_path in imagesPath:
    image = read_image_bgr(full_path)
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    
    # process image
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    
    # correct for image scale
    boxes /= scale
    
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        color = label_color(0)
        
        b = box.astype(int)
        print(b)

        centers.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))

        # write the predicted images
        img = cv2.imread(full_path)
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        name = full_path.split("\\")[-1]
        changed_path = predictionFolder + "\\" + name
        cv2.imwrite(changed_path, img)
        
        break

print("Writing the data to a file")
with open('centers.txt', 'w') as f:
    for item in centers:
        f.write("{0}\n".format(item))