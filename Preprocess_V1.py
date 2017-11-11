#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:14:21 2017

@author: prithvi
"""

import yaml
import numpy as np
from scipy import ndimage
from scipy.misc import imsave
import pickle

##############################################################################

with open("./train_data/train.yaml", 'r') as stream:
    try:
        img_index = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

idx = 0

for entry in img_index:
    path = './train_data'+entry['path'].split('.')[1]+'.png'
    im_array = ndimage.imread(path)
    mask = np.zeros([im_array.shape[0],im_array.shape[1],3])
    boxes = entry['boxes']
    for box in boxes:
        x_min = int(box['x_min'])
        x_max = int(box['x_max'])
        y_min = int(box['y_min'])
        y_max = int(box['y_max'])
        if('Green' in box['label']):
            mask[y_min:y_max,x_min:x_max,0] = 1.0
        elif('Yellow' in box['label']):
            mask[y_min:y_max,x_min:x_max,1] = 1.0
        elif('Red' in box['label']):
            mask[y_min:y_max,x_min:x_max,2] = 1.0
        
    path = path.replace('bag','bag_mask_enhanced')
    imsave(path,mask)
    print(idx)
    idx +=1
#############################################################################

with open("./test_data/test.yaml", 'r') as stream:
    try:
        img_index = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

idx = 0

for entry in img_index:
    path = './test_data/rgb/test/'+entry['path'].split('/')[-1]
    im_array = ndimage.imread(path)
    mask = np.zeros([im_array.shape[0],im_array.shape[1],3])
    boxes = entry['boxes']
    for box in boxes:
        x_min = int(box['x_min'])
        x_max = int(box['x_max'])
        y_min = int(box['y_min'])
        y_max = int(box['y_max'])
        if('Green' in box['label']):
            mask[y_min:y_max,x_min:x_max,0] = 1.0
        elif('Yellow' in box['label']):
            mask[y_min:y_max,x_min:x_max,1] = 1.0
        elif('Red' in box['label']):
            mask[y_min:y_max,x_min:x_max,2] = 1.0
    mask_path = './test_data/rgb/test_mask_enhanced/'+entry['path'].split('/')[-1]
    imsave(mask_path,mask)
    print(idx)
    idx +=1