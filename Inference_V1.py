#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import numpy as np
from keras.models import Model,load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, UpSampling2D, Lambda
from keras.layers.convolutional import Cropping2D
from keras.layers.core import Reshape
from keras.optimizers import Adam
from keras.preprocessing import image
from keras import backend as K
from keras import applications
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
import os
from sklearn.metrics import log_loss
from scipy.ndimage.measurements import label
import tensorflow as tf
from scipy import ndimage



import scipy

import matplotlib.pyplot as plt
import cv2 

import yaml
import random

img_rows,img_cols = 512,512
mask_rows,mask_cols = 512,512

def return_bound_rect(img,img_to_imprint,color):
    img = img*255
    img = img.astype(np.uint8)
    ret,thresh = cv2.threshold(img,127,255,0)
    rim,contours,hierarchy = cv2.findContours(thresh, 1, 2)
    img2 = np.zeros_like(img)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        #print(w,h)
        #if(w<200&h<200):
        img2[y:y+h,x:x+w] = 1.0
        img_to_imprint = cv2.rectangle(img_to_imprint,(x,y),(x+w,y+h),color,3)
    
    return img_to_imprint,img2

def categorical_crossentropy_with_weights(y_true, y_pred):
    class_weights = tf.constant(np.array([1.,1.,1.,0.001], dtype=np.float32))
    weighted_y_true = tf.multiply(y_true, class_weights)
    
    return K.categorical_crossentropy(weighted_y_true, y_pred)

nnet = load_model('epoch_4d_aug_390.h5', custom_objects={'categorical_crossentropy_with_weights': categorical_crossentropy_with_weights})

def preprocess_im_correct(image_1,flipper,tc,bc,lc,rc):
    
    re_img = image_1[tc:image_1.shape[0]-bc,lc:image_1.shape[1]-rc,:]
    if(flipper==1):
        re_img = np.fliplr(re_img)
    re_img = scipy.misc.imresize(re_img, [img_rows,img_cols,3])
    re_img = re_img/255.

    return re_img

def draw_traffic_light_box(img):
    
    pred_mask = nnet.predict(img)
    
    pred_mask = np.reshape(pred_mask,(512,512,4))
    
    #Green mask
    green_mask = pred_mask[:,:,0]
    green_mask[green_mask<0.5] = 0.0
    green_mask[green_mask>=0.5] = 1.0
    
    #Red mask
    red_mask = pred_mask[:,:,1]
    red_mask[red_mask<0.5] = 0.0
    red_mask[red_mask>=0.5] = 1.0


    #Yellow mask
    yellow_mask = pred_mask[:,:,2]
    yellow_mask[yellow_mask<0.5] = 0.0
    yellow_mask[yellow_mask>=0.5] = 1.0

    boundary_mask = red_mask+green_mask+yellow_mask
    
    boundary_box,_ = return_bound_rect(boundary_mask,np.copy(img[0]),(200,0,0))
    #boundary_box,_ = return_bound_rect(yellow_mask,boundary_box,(0,0,200))
    #boundary_box,_ = return_bound_rect(green_mask,boundary_box,(0,200,0))
    
    return boundary_box

def read_image(image_path):
    
    #cv2.imread(image_path,0)
    im_array = ndimage.imread(image_path)
    
    return im_array
    
def prepare_im(img_path):
    
    img = read_image(img_path)
    
    img = preprocess_im_correct(img,0,0,0,0,0)
    
    img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
    
    return img

inference_images = img_paths = ['1509823937_004.jpg','1509798552_160.jpg','1509823939_659.jpg','1509820260_008.jpg','1509823939_190.jpg','26580.png','1509823933_595.jpg','1509815712_170.jpg','1509815723_546.jpg']

#inference_images = os.listdir('./real-images')

for imag in inference_images:
    
    im_to_predict = prepare_im(imag)

    bbox = draw_traffic_light_box(im_to_predict)
    plt.imshow(bbox)
    plt.show()
    plt.close()
    

    