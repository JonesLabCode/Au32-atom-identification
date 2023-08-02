# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:04:36 2021

@author: carri
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from  scipy.signal import argrelextrema
from scipy import ndimage, misc
import os
from glob import glob
from utils import readImages_tif

def localization(stack,thresh,size):
    if stack == '1-4':
        image_dir = 'original stacks/'+stack  + ' stacks'
        image_names = readImages_tif(image_dir)

    if stack == '1-5':
        image_dir = 'original stacks/' + stack + ' stacks'
        image_names = readImages_tif(image_dir)
        thresh=128
        size =10

    if stack == '1-2':
        image_dir = 'original stacks/'+stack  + ' stacks'
        image_names = readImages_tif(image_dir)
        size = 9
        thresh = 174

    if stack =='1-3':
        image_dir = 'original stacks/' + stack + ' stacks'
        image_names = readImages_tif(image_dir)
        size = 7
        thresh = 145
    if stack =='1-1':
        image_dir = 'original stacks/' + stack + ' stacks'
        image_names = readImages_tif(image_dir)
        size = 11
        thresh = 120
    if stack=='2-3':
        image_dir = 'original stacks/' + stack + ' stacks'
        image_names = readImages_tif(image_dir)
        size = 5
        thresh = 130



    for i in range(len(image_names)):
        print('marking frame',i)
        if stack=='1-3'and i>=3:
            thresh=145
        image = cv2.imread(image_names[i])
        img = image
        img = cv2.GaussianBlur(image, (3,3), sigmaX=3)
        gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        max_img = ndimage.maximum_filter(gray, size=size)

        below = max_img < thresh
        max_img[below] = 0

        max_pixel = max_img==gray
        mark = np.zeros_like(gray)
        mark[max_pixel] = 255


        pt = []
        last_row = 0
        last_col = 0
        for row in range(mark.shape[0]):
            for col in range(mark.shape[1]):
                if mark[row,col]==255:
                    if (last_row - row)< -0 or (last_col - col) <-2:
                        pt.append((col,row))
                        circles = cv2.circle(img,(col,row),1,(0,0,255),1)
                        last_row = row
                        last_col = col

        xyfeatures = np.zeros((len(pt),2))
        for pt_i in range(len(pt)):
            xyfeatures[pt_i,0] = pt[pt_i][0]
            xyfeatures[pt_i,1] = pt[pt_i][1]
        np.save('marked frames/{} marked frames/pt_location_frame{}.npy'.format(stack,i),xyfeatures)
        cv2.imwrite('marked frames/{} marked frames/frame{} local maximum s{} t{}.tif'.format(stack,i,size,thresh),circles)


#stack = '1-3'
#localization(stack,174,9)