#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 20:43:46 2017

@author: prithvi
"""
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

#patch_img_size = (100,100)

model = ResNet50(weights='imagenet')

def return_preds(img):
    
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    
    preds = model.predict(x)
    
    return preds





