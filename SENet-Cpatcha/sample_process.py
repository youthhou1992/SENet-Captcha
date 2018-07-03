# -*- coding: utf-8 -*-
"""
Created on Sat May 12 15:16:14 2018

@author: songweinan
"""
from PIL import Image
import numpy as np
import os
import pickle
from itertools import groupby
import math
import cv2
import os
from glob import glob
import random
import string
import re

PATH='/home/youth/DL/SENet-Cpatcha/imgs/val'
path=PATH+'/*'
num=0
node=0
for img_full_name in glob(path):
    num+=1
   # if num<280000:
        #continue
    img = cv2.imdecode(np.fromfile(img_full_name, dtype=np.uint8), -1)
    print(img_full_name)
#    if img.shape[1]==192 and img.shape[2]==64:
#        num+=1
#        if num==node + 10000:
#            #print('1')
#            print(num//10000)
#            node=num
#        continue
    img = cv2.resize(img, (192, 64))
    shape=img.shape
    if len(shape)==2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #print(img_full_name)
    
    cv2.imencode('.png', img)[1].tofile(img_full_name)
    
    if num==node + 10000:
        print(num//10000)
        node=num
print("1")