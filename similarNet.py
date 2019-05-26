#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 23:13:07 2019

@author: chandravanshi
"""
### IMPORTING LIBRARIES
import pandas as pd
import numpy as np

import keras
from keras_retinanet import models
from keras_retinanet.models import load_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import keras_retinanet
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


from keras.preprocessing.image import ImageDataGenerator

import cv2
import tensorflow as tf
# keras libraries
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, Activation, BatchNormalization, Dropout, ZeroPadding2D
from keras.layers.merge import add
from keras.models import Model, load_model
from keras import backend as k
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import img_to_array
from keras.layers import concatenate, Subtract,Concatenate, subtract
### Initializing the model
model = keras_retinanet.models.backbone('resnet50').retinanet(num_classes=1)

model.compile(
    loss={
        'regression'    : keras_retinanet.losses.smooth_l1(),
        'classification': keras_retinanet.losses.focal()
    },
    optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
)

# loading the pretrained model
model = models.load_model("resnet_coco.h5", backbone_name='resnet50')
print(model.summary())

# proceed with the similarNet
# Following are the layers that we're gonna extract
low='res2a_relu'
mid='res3a_relu'
high='res4f_relu'
layers = [low, mid, high]


# we'll input an image and will try to extract some of the hidden layers
path2image1 = "images/test2.jpg"
path2image2 = "images/test1.jpg"

def returnProcessedImage(path2image): 
    # preprocessing the image to be fed into the model
    image = read_image_bgr(path2image)
    image = preprocess_image(image)
    image, scale = resize_image(image)
    image = np.expand_dims(image, axis=0)
    return image

image1 = returnProcessedImage(path2image1)
image2 = returnProcessedImage(path2image2)

def returnHiddenLayers(layers_name, image):
    outputs = [model.get_layer(layer_name).output for layer_name in layers_name]
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=outputs)
    intermediate_output = intermediate_layer_model.predict(image)
    return intermediate_output

intermediate_output1 = returnHiddenLayers(layers, image1)
intermediate_output2 = returnHiddenLayers(layers, image2)

# let's look at the shape of outputs
for i, output in enumerate(intermediate_output1):
    print("shape of layer-> {0} for image1 is: {1}".format(layers[i], np.shape(output)))
print("-------------------------------from keras.layers import concatenate, Subtract,Concatenate----------------")
for i, output in enumerate(intermediate_output2):
    print("shape of layer-> {0} for image2 is: {1}".format(layers[i], np.shape(output)))


# for similarity network we'll be feeding the hidden layers of the RetinaNet
# hidden layer corresponding to image1
hidden_layer1a = intermediate_output1[0]
hidden_layer2a = intermediate_output1[1]
hidden_layer3a = intermediate_output1[2]
# hidden layer corresponding to image2
hidden_layer1b = intermediate_output2[0]
hidden_layer2b = intermediate_output2[1]
hidden_layer3b = intermediate_output2[2]

# we need to pad the input images by 1, so that if we divide them with 2 they'll be
# consistent,  # (1, W, H, C) -> # (W, H, C)

def remove_dimensions(array):
    """ It converts (1, W, H, C) -> (W, H, C) given the array is of 4 dimension
    """
    shapes = np.shape(array)
    if len(shapes) == 4:
        array = np.reshape(array, (shapes[1], shapes[2], shapes[3]))
    return array

# remove the first dimension of hidden layers of image1
hidden_layer1a = remove_dimensions(hidden_layer1a)
hidden_layer2a = remove_dimensions(hidden_layer2a)
hidden_layer3a = remove_dimensions(hidden_layer3a)

# remove the first dimension of hidden layers of image2
hidden_layer1b = remove_dimensions(hidden_layer1b)
hidden_layer2b = remove_dimensions(hidden_layer2b)
hidden_layer3b = remove_dimensions(hidden_layer3b)


def add_dimensions(array):
    return np.expand_dims(array, axis=0)

"""
def pad_tensor(tensor):
    temp = tensor # for temporary assignments
    shape = np.shape(tensor)
    print("Pad tensor is called! shape of tensor is: ", shape)
    if shape[1] % 2 != 0 and shape[2] % 2 != 0:
        print("both are odd")
        temp = pad_tensor_row(tensor)
        temp = pad_tensor_col(temp)
        print("after padding: ", np.shape(temp))
    elif shape[1] % 2 != 0:
        print("row is odd")
        temp = pad_tensor_row(tensor)
    elif shape[2] % 2 != 0:
        print("col is odd")
        temp = pad_tensor_col(tensor)
    return temp
"""
def pad_tensor(tensor):
    temp = tensor # for temporary assignments
    shape = np.shape(tensor)
    print("Pad tensor is called! shape of tensor is: ", shape)
    if shape[1] % 2 != 0 and shape[2] % 2 != 0:
        print("both are odd")
        temp = ZeroPadding2D(padding=((0, 1),(0, 1)))(tensor)
        print("after padding: ", np.shape(temp))
    elif shape[1] % 2 != 0:
        print("row is odd")
        temp = ZeroPadding2D(padding=((0, 1), (0, 0)))(tensor)
        print("after padding: ", np.shape(temp))
    elif shape[2] % 2 != 0:
        print("col is odd")
        temp = ZeroPadding2D(padding=((0, 0), (0, 1)))(tensor)
        print("after padding: ", np.shape(temp))
    return temp


def returnConcatedConv(input1, input2, input3):
    
    # we'll need to expand the 3 channel to 512
    i1 = Conv2D(512, kernel_size=(1, 1))(input1)
    o1 = MaxPooling2D(pool_size=4)(i1)
    o1 = pad_tensor(o1)
    # we'll need to apply maxpooling on second layer
    o2 = MaxPooling2D()(input2)
    o2 = pad_tensor(o2)
    # we'll need to change the number of filters in case of third input
    o3 = Conv2D(512, kernel_size=(1, 1))(input3)
    o3 = pad_tensor(o3)
    print("--------------------------------------")
    print("Shape of O1: ", np.shape(o1))
    print("Shape of O2: ", np.shape(o2))
    print("Shape of O3: ", np.shape(o3))
    print("--------------------------------------")
    concated_layers = concatenate([o1, o2, o3])
    print("Shape of Concated Layers: ", np.shape(concated_layers))
    print("--------------------------------------")
    return concated_layers

def returnDenseLayer(concated_layers):
    x = Conv2D(filters=512, kernel_size=(3, 3), padding="same")(concated_layers)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)
    x = Conv2D(filters=1024, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    return Dense(units=64, activation="relu")(x)
    

def similarModel(input1a, input2a, input3a, input1b, input2b, input3b):
    concated_layer1 = returnConcatedConv(input1a, input2a, input3a)
    concated_layer2 = returnConcatedConv(input1b, input2b, input3b)
        
    print("Shape of Concated Layers 1: ", np.shape(concated_layer1))
    print("Shape of Concated Layers 2: ", np.shape(concated_layer2))
    
    # apply some convolutional layers
    
    dense1 = returnDenseLayer(concated_layer1)
    dense2 = returnDenseLayer(concated_layer2)
    
    print("Shape of Dense1: ", np.shape(dense1))
    print("shape of Dense2: ", np.shape(dense2))
    
    subtracted = subtract([dense1, dense2])
    x = Dense(units=32, activation="relu")(subtracted)
    out = Dense(units=1, activation="sigmoid")(x)
    
    model = Model(inputs=[input1a, input2a, input3a, input1b, input2b, input3b], outputs=[out])
    return model
    
    
    
input1a = Input(shape=(hidden_layer1a.shape[0], hidden_layer1a.shape[1], hidden_layer1a.shape[2]))
input2a = Input(shape=(hidden_layer2a.shape[0], hidden_layer2a.shape[1], hidden_layer2a.shape[2]))
input3a = Input(shape=(hidden_layer3a.shape[0], hidden_layer3a.shape[1], hidden_layer3a.shape[2]))

input1b = Input(shape=(hidden_layer1b.shape[0], hidden_layer1b.shape[1], hidden_layer1b.shape[2]))
input2b = Input(shape=(hidden_layer2b.shape[0], hidden_layer2b.shape[1], hidden_layer2b.shape[2]))
input3b = Input(shape=(hidden_layer3b.shape[0], hidden_layer3b.shape[1], hidden_layer3b.shape[2]))

input1a = Input(shape=(3,))
input2a = Input(shape=(3,))
input3a = Input(shape=(3,))
input1b = Input(shape=(3,))
input2b = Input(shape=(3,))
input3b = Input(shape=(3,))

    
    
similar_model = similarModel(input1a, input2a, input3a, input1b, input2b, input3b)
similar_model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
print(similar_model.summary())


############# HOW WILL YOU TRAIN THE MODEL #######################






















