#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 09:39:16 2018

@author: youth
"""

import numpy as np
#import pandas as pd
import cv2
import os

def draw_boxs(path, name):
    #img_path = '/home/youth/Downloads/train_1000/image_1000/TB1..FLLXXXXXbCXpXXunYpLFXX.png'
    #label_path = '/home/youth/Downloads/train_1000/txt_1000/TB1..FLLXXXXXbCXpXXunYpLFXX.txt'
    img_dir = 'image_1000/%s.png'%name
    label_dir = 'txt_1000/%s.txt'%name
    
    img_path = os.path.join(path, img_dir)
    label_path = os.path.join(path, label_dir)
    print img_path, label_path
    
    if not os.path.exists(img_path):
        print 'image doesn\'t exist'
        return
    if not os.path.exists(label_path):
        print 'label file doesn\'t exist'
        return
    
    img = cv2.imread(img_path, 1)
    with open(label_path, 'r') as f:
        #cood = []
        for line in f.readlines():
            cood = line.split(',')[:8]
            x = [int(float(cood[i])) for i in [0,2,4,6]]
            y = [int(float(cood[i])) for i in [1,3,5,7]]
            point = [[a,b] for a,b in zip(x,y)]
            point = np.array(point)          
            point = point.reshape((-1,1,2))
    #        print np.shape(point)
            cv2.polylines(img, [point], True, (0,0,0),5)
    return img

if __name__ == '__main__':
    file_path = '/home/youth/Downloads/train_1000'
    name = 'TB1..FLLXXXXXbCXpXXunYpLFXX'
    img = draw_boxs(file_path, name)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)