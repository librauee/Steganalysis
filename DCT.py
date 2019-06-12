# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:10:43 2019

@author: Administrator
"""


import cv2
import numpy as np


img = cv2.imread('a.png',0)
img1 = img.astype('float')
 
def dct(m):
    m = np.float32(m)/255.0
    return cv2.dct(m)*255
#print(dct(img1).shape)
new_dct=dct(img1)
after_dct=[]
for i in range(len(new_dct)):
    for j in range(len(new_dct[0])):
        after_dct.append(int(new_dct[i][j]))
#print(new_dct)
#new_dct=new_dct.reshape(-1,1)
#print(len(after_dct))
#print(after_dct[:600])


    
