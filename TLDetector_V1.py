#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:27:00 2017

@author: prithvi
"""


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

with open("./train_data/train.yaml", 'r') as stream:
    try:
        img_index = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

with open("./test_data/test.yaml", 'r') as stream:
    try:
        valid_img_index = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


batch_size = 4


    


def categorical_crossentropy_with_weights(y_true, y_pred):
    class_weights = tf.constant(np.array([1.,1.,1.,0.000001], dtype=np.float32))
    weighted_y_true = tf.multiply(y_true, class_weights)
    
    return K.categorical_crossentropy(weighted_y_true, y_pred)




def get_unet_512(input_shape=(img_rows, img_cols, 3),
                 num_classes=4):
    inputs = Input(shape=input_shape)
    # 512

    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    # 512

    classify = Conv2D(num_classes, (1, 1), activation='relu')(up0a)
    #norm = Lambda(normalizer)(classify)
    #classify = Activation('softmax')(classify)
    classify = Reshape([mask_rows*mask_cols,4],name='Rsp')(classify)
    classify = Activation('softmax')(classify)
    #classify = Reshape([mask_rows*mask_cols*4],name='Rsp2')(classify)
    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=RMSprop(), loss=categorical_crossentropy_with_weights)

    return model    


def read_image(image_path):
    
    #cv2.imread(image_path,0)
    im_array = ndimage.imread(image_path)
    
    return im_array

def read_mask(image_path):
    
    #cv2.imread(image_path,0)
    im_array = ndimage.imread(image_path)
    
    return im_array


def preprocess_im_correct(image_1,flipper,tc,bc,lc,rc):
    
    re_img = image_1[tc:image_1.shape[0]-bc,lc:image_1.shape[1]-rc,:]
    if(flipper==1):
        re_img = np.fliplr(re_img)
    re_img = scipy.misc.imresize(re_img, [img_rows,img_cols,3])
    re_img = re_img/255.

    return re_img

    
def preprocess_mask(image_1,flipper,tc,bc,lc,rc):
    
    re_img = image_1[tc:image_1.shape[0]-bc,lc:image_1.shape[1]-rc,:]

    if(flipper==1):
        re_img = np.fliplr(re_img)    

    re_img_2 = np.copy(re_img)
    
    re_img_2[re_img<127.0] = 0.0

    re_img_2[re_img>=127.0] = 1.0
    
    re_img_2 = scipy.misc.imresize(re_img_2, [mask_rows,mask_cols])
    
    re_img_3 = ((re_img_2[:,:,0]+re_img_2[:,:,1]+re_img_2[:,:,2]) + 1)%2
    
    re_img_2 = np.reshape(re_img_2,(mask_rows*mask_cols,3))
    
    re_img_3 = np.reshape(re_img_3,(mask_rows*mask_cols))

    return re_img_2, re_img_3

    
def gen_train_image_batch():
    
    #Generator to keep outputting batches at random
    #Pick random images from the test set
    #Preprocess the images
    
    train_x = np.zeros([batch_size,img_rows,img_cols,3])
    train_y = np.zeros([batch_size,mask_rows*mask_cols,4])
    while(1):
        for i in range(batch_size):
            im = random.choice(img_index)
            img_path = './train_data'+im['path'].split('.')[1]+'.png'
            mask_path = img_path.replace('bag','bag_mask_enhanced')
            img = read_image(img_path)
            mask = read_mask(mask_path)
            im_w = img.shape[1]
            im_h = img.shape[0]
    
            #Crop the image randomly to train on patches of the image; A kind of data augmentation.
            top_cut = random.choice(range(int(im_h/3)))
            bottom_cut = random.choice(range(int(im_h/3)))
            left_cut = random.choice(range(int(im_w/3)))
            right_cut = random.choice(range(int(im_w/3)))
            flipper = random.choice([0,1])
            train_x[i,:,:,:] = preprocess_im_correct(img[:,:,:],flipper,top_cut,bottom_cut,left_cut,right_cut)
            train_y[i,:,:3],train_y[i,:,3] = preprocess_mask(mask,flipper,top_cut,bottom_cut,left_cut,right_cut)
            i +=1
            
    
        yield train_x, train_y

def gen_test_image_batch():
    
    train_x = np.zeros([batch_size,img_rows,img_cols,3])
    train_y = np.zeros([batch_size,mask_rows*mask_cols,4])
    batch_size_met = 0
    while(batch_size_met<batch_size):
        #random.seed(42)
        im = random.choice(valid_img_index)
        img_path = './test_data/rgb/test/'+im['path'].split('/')[-1]
        mask_path = './test_data/rgb/test_mask_enhanced/'+im['path'].split('/')[-1]
        img = read_image(img_path)
        mask = read_mask(mask_path)
        #plt.imshow(img)
        #plt.imshow(mask)
        im_w = img.shape[1]
        im_h = img.shape[0]
        top_cut = random.choice(range(int(im_h/3)))
        bottom_cut = random.choice(range(int(im_h/3)))
        left_cut = random.choice(range(int(im_w/3)))
        right_cut = random.choice(range(int(im_w/3)))
        flipper = random.choice([0,1])
        train_x[batch_size_met,:,:,:] = preprocess_im_correct(img[:,:,:],flipper,top_cut,bottom_cut,left_cut,right_cut)
        train_y[batch_size_met,:,:3],train_y[batch_size_met,:,3] = preprocess_mask(mask,flipper,top_cut,bottom_cut,left_cut,right_cut)
        batch_size_met +=1
    
    yield train_x,train_y


#Initialize a neural net

nnet = get_unet_512()

#read the train images and masks by batch

nnet.fit_generator(generator = gen_train_image_batch(), steps_per_epoch=1000,epochs=10)
