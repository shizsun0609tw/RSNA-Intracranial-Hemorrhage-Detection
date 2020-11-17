# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:15:49 2020

@author: ctlian
"""

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import os

def min_max(img, l, w):
    return (img - (l-w/2))/w

def window_image(dcm, window_center, window_width, resample):
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)

    # resampling
    # Determine current pixel spacing
    if resample:
        new_spacing = [1, 1]
        spacing = np.array(dcm.PixelSpacing, dtype=np.float32)
        
        resize_factor = spacing / new_spacing
        new_real_shape = img.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / img.shape
        new_spacing = spacing / real_resize_factor
        
        img = scipy.ndimage.interpolation.zoom(img, real_resize_factor, mode='nearest')

    return img

def bsb_window(path, resample=False):
    dcm = pydicom.dcmread(path)
    
    #plt.figure(num=path.split(sep='.')[0],figsize=(8,8))
    
    brain_img = window_image(dcm , 40, 80, resample)
    #plt.subplot(1,3,1) 
    #plt.imshow(brain_img)
    subdural_img = window_image(dcm , 80, 200, resample)
    #plt.subplot(1,3,2) 
    #plt.imshow(subdural_img)
    bone_img = window_image(dcm , 600, 2000, resample)
    #plt.subplot(1,3,3) 
    #plt.imshow(bone_img)
    #plt.show()
    brain_img = min_max(brain_img, 40, 80)
    subdural_img = min_max(subdural_img, 80, 200)
    bone_img = min_max(bone_img, 600, 2000)
    
    img = np.array([brain_img, subdural_img, bone_img]).transpose(1,2,0)
    return img
    
def main():
    type_list = ['epidural','healthy','intraparenchymal','intraventricular','subarachnoid','subdural']
    os.mkdir('image')
    for type_name in type_list:
        os.mkdir('image/' + type_name)
    
    for i in range(len(type_list)):
        #data = []
        #target = []
        path = 'TrainingData/' + type_list[i] + '/'
        print(path)
        file_list = os.listdir(path)
        for file in file_list:
            file_name = file.split(sep='.')[0]
            file_path = path + file
            img = bsb_window(file_path)
            #img_transpose = img.transpose(2,0,1)
            
            #data.append(np.array(img_transpose))
            #target.append(i)
            #plt.imshow(img)
            plt.imsave('image/' + type_list[i] + '/' + file_name + '.png', img)
            
        #np.save('data_' + type_list[i] + '.npy', data)
        #np.save('target_' + type_list[i] + '.npy', target)
    
def test_main():
    os.mkdir('testImage')
    path = 'TestingData/'
    
    file_list = os.listdir(path)
    for file in file_list:
        file_name = file.split(sep='.')[0]
        file_path = path + file
        img = bsb_window(file_path)
        
        plt.imsave('testImage/' + file_name + '.png', img)
            
    
    
if __name__ == '__main__':
    test_main()