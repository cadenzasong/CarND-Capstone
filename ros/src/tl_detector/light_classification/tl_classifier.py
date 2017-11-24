#==============================================================================
# 
# License for using the Keras pretrained model in Line 71
# 
# MIT License
# 
# Copyright (c) 2016 François Chollet
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#==============================================================================

from styx_msgs.msg import TrafficLight
from keras.models import Model,load_model
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.applications.inception_v3 import preprocess_input
import rospy
import scipy
import cv2
import time
from scipy.misc import imsave
import os

def categorical_crossentropy_with_weights(y_true, y_pred):
    class_weights = tf.constant(np.array([1.,1.,0.001], dtype=np.float32))
    weighted_y_true = tf.multiply(y_true, class_weights)
    
    return K.categorical_crossentropy(weighted_y_true, y_pred)
    
def preprocess_im_correct(image_1,flipper,tc,bc,lc,rc):
    
    re_img = image_1[tc:image_1.shape[0]-bc,lc:image_1.shape[1]-rc,:]
    if(flipper==1):
        re_img = np.fliplr(re_img)
    re_img = scipy.misc.imresize(re_img, [512,512,3])
    re_img = re_img/255.

    return re_img

def prepare_im_test(img):
    
    img = preprocess_im_correct(img,0,0,0,0,0)
    
    img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
    
    return img
    
class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.unet = load_model('./../../../weights/concise_weights.19.h5', custom_objects={'categorical_crossentropy_with_weights': categorical_crossentropy_with_weights})
        self.unet._make_predict_function()
        self.graph_unet = tf.get_default_graph()
        self.Resnet = load_model('./../../../weights/InceptionV3_keras.h5')
        self.Resnet._make_predict_function()
        self.graph_Resnet = tf.get_default_graph()
        self.light_color_dict = {'Red':TrafficLight.RED,'Green':TrafficLight.GREEN,'No light':TrafficLight.UNKNOWN}

        pass
    
    def return_preds(self,img):
    
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        
        with self.graph_Resnet.as_default():
            preds = self.Resnet.predict(x)
        
        #preds = self.Resnet.predict(x)
        
        return preds
    def return_best_rect(self,img,img_to_imprint):
        img = img*255
        img = img.astype(np.uint8)
        ret,thresh = cv2.threshold(img,127,255,0)
        _,contours,hierarchy = cv2.findContours(thresh, 1, 2)
        best_prob = 0.1
        best_rect = [0,0,0,0]
        light_found = False
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if(w>=10 and h>=20):
                patch_image = img_to_imprint[y:y+h,x:x+w,:]
                patch_image = scipy.misc.imresize(patch_image, [224,224,3])
                patch_image = patch_image.astype(np.float16)
    
                tl_prob = self.return_preds(patch_image)[0,920]
    
                if(tl_prob>=best_prob):
                    best_rect = [x,y,w,h] 
                    best_prob = tl_prob
                    light_found = True
                    
        return best_rect,light_found    
    def detect_traffic_light_color(self,img):
    
        threshold = 0.95
        
        with self.graph_unet.as_default():
            pred_mask = self.unet.predict(img)
        

    
        pred_mask = np.reshape(pred_mask,(512,512,3))
        
        #Green mask
        green_mask = pred_mask[:,:,0]

        green_mask[green_mask<threshold] = 0.0
        green_mask[green_mask>=threshold] = 1.0
        
        #Red mask
        red_mask = pred_mask[:,:,1]
        
        red_mask[red_mask<threshold] = 0.0
        red_mask[red_mask>=threshold] = 1.0
    
        boundary_mask = red_mask+green_mask
        
        [x,y,w,h],lf = self.return_best_rect(boundary_mask,np.copy(img[0]))
        class_mask = np.argmax(pred_mask,axis=2)[y:y+h,x:x+w]
        
        green_tl = np.sum(class_mask==0)
        red_tl = np.sum(class_mask==1)
    
        
        
        if(lf):
            if(green_tl>red_tl):
                light_color = 'Green'
            else:
                light_color = 'Red'
        else:
            light_color = 'No light'
    
        return light_color
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        im_arr = np.asarray(image)
        im_arr = prepare_im_test(im_arr)
        lc = self.detect_traffic_light_color(im_arr)

        
        return self.light_color_dict[lc]
