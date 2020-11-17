import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def read(path):
    return cv2.imread(path)

def noise_reduction1(img):
    return cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

def noise_reduction2(img):
    blurred = cv2.GaussianBlur(img,(5,5),0)
    median = cv2.medianBlur(blurred, 5)
    kernel = np.ones((5,5),np.float32) / 25
    lowpass_filter = cv2.filter2D(median,-1,kernel)

    return lowpass_filter

def enhancemet(img):
    channel0, channel1, channel2 = cv2.split(img)

    channel0 = cv2.equalizeHist(channel0)
    channel1 = cv2.equalizeHist(channel1)
    channel2 = cv2.equalizeHist(channel2)

    img = cv2.merge([channel0, channel1, channel2])

    return img

def Otsu(img):
    channel0, channel1, channel2 = cv2.split(img)

    ret, channel0 = cv2.threshold(channel0, 0, 255, cv2.THRESH_OTSU)
    ret, channel1 = cv2.threshold(channel1, 0, 255, cv2.THRESH_OTSU)
    ret, channel2 = cv2.threshold(channel2, 0, 255, cv2.THRESH_OTSU)
    
    img = cv2.merge([channel0, channel1, channel2])

    return img

def clip(img):
    height, width, channels = img.shape
    
    left = width
    top = height
    right = 0
    bottom = 0

    for i in range(height):
        for j in range(width):
            for k in range(channels):
                if img[i][j][k] != 0:
                    left = min(j, left)
                    right = max(j, right)
                    top = min(i, top)
                    bottom = max(i, bottom)

    if top >= bottom or left >= right:
        top = 0
        bottom = height
        left = 0
        right = width

    return img[top:bottom, left:right]

def save(img, path):
    cv2.imwrite(path, img)

def run(read_path, save_path, class_path):
    for i in range(len(class_path)):
        files = os.listdir(read_path + class_path[i])
        
        try:
            os.mkdir(save_path + class_path[i])
        except:
            pass

        for file in files:
            origin_img = read(read_path + class_path[i] + file)
            clip_img = clip(origin_img)
            # filter_img = noise_reduction1(clip_img)
            filter_img = noise_reduction2(clip_img)
            # filter_img1 = enhancemet(filter_img)
            # filter_img2 = Otsu(filter_img)
            save(filter_img, save_path + class_path[i] + file)

            print("Save Img: " + save_path + class_path[i] + file + ", Shape: " + str(filter_img.shape[0]) + ', ' + str(filter_img.shape[1]))


################################################ main ################################################
read_path = 'test_image/'
save_path = 'test_filter_image/'

class_path = ['']

run(read_path, save_path, class_path)