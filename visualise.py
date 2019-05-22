# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:31:46 2019

@author: Kishan Kumar
"""

import matplotlib.pyplot as plt
import pandas as pd
import cv2

df = pd.read_csv('new_annotations.csv')
path2images = df.iloc[:,0:1].values
boundingBoxes= df.iloc[:,1:5].values

for path, bounding in zip(path2images, boundingBoxes):
    img = cv2.imread(path[0])
    x1=bounding[0] 
    x2=bounding[2] 
    y1=bounding[1] 
    y2=bounding[3]
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()