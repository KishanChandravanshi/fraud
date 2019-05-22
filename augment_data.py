# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:45:29 2019

@author: Kishan Kumar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import imgaug as ia
from imgaug import augmenters as iaa 

df = pd.read_csv('final_annotation.csv')
path2images = df.iloc[:,0:1].values
boundingBoxes= df.iloc[:,1:5].values

savedir = "Final Augmented Data\\"

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

for path, bounding in zip(path2images, boundingBoxes):
    # read the image into a numpy array using imageio
    for i in range(8):
        print("working on {} image, please wait...".format(path[0].split('\\')[-1]))
        counter = 0  # because we'll be creating three custom images
        img = imageio.imread(path[0])
        # now read the bounding boxes
        bbx = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=bounding[0], x2=bounding[2], y1=bounding[1], y2=bounding[3])], shape=img.shape)
        
        #ia.imshow(bbx.draw_on_image(img, thickness=2))
        
        # let's apply augmentation
        seq = iaa.Sequential([
        iaa.Affine(rotate=[-90, -180, -270, -360]),
        #sometimes(iaa.Affine(translate_percent={"x": 0.1}, scale=0.8)),
        sometimes(iaa.AdditiveGaussianNoise(scale=(1, 2))),
        sometimes(iaa.Crop(percent=(0, 0.05)))
        #sometimes(iaa.OneOf([
        #            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
        #            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
        #            iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
        #        ])),
        #sometimes(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))) # sharpen images
        #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
        #sometimes(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
        ])
        
        seq_det = seq.to_deterministic()
        # give an image
        image_aug = seq_det.augment_image(img)
        bbs_aug = seq_det.augment_bounding_boxes([bbx])[0]
    
        x1 = bbs_aug.bounding_boxes[0].x1_int
        x2 = bbs_aug.bounding_boxes[0].x2_int
        y1 = bbs_aug.bounding_boxes[0].y1_int
        y2 = bbs_aug.bounding_boxes[0].y2_int
        
        #ia.imshow(bbs_aug.draw_on_image(image_aug, thickness=2))
        
        #print("original Bounding box",bounding)
        #print("changed",bbs_aug)
        # it'll be good to save the augmented image and the corresponding bounding box in some different folder
        temp = path[0].split('\\')[-1]
        temp = temp.split('.')[0]
        newpath = savedir + temp + "aug" + str(i) + ".jpg" 
        imageio.imwrite(newpath, image_aug)
        
        with open("final_new_annotations.txt", 'a') as file:
            things_to_write = newpath + "," + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + "," + "vortex_center" + "\n"
            file.write(things_to_write)
        