# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 23:33:18 2023

@author: marie
"""
# load libraries
from ultralytics import YOLO
import cv2
import os
import numpy as np

# load model and test images
model_path = r'runs\segment\train\weights\best.pt'
test_im =  r'images\test'
test_lb =  r'labels\test'
out_path = r'predictions'

model = YOLO(model_path)
image_list = os.listdir(test_im)



def create_mask(true_segment, pred_segment):
    mask = np.zeros([height,width])
    true_mask = cv2.drawContours(mask, [true_segment], -1, 255, cv2.FILLED, 1) 
    pred_mask = cv2.drawContours(mask, [pred_segment], -1, 255, cv2.FILLED, 1) 
    true_mask[true_mask>0] = 1
    pred_mask[pred_mask>0] = 1
    return true_mask, pred_mask

# get true segment
with open(label_path,'r') as f:
    line = f.readline()
    line = [float(num) for num in line.split(' ')]
    segment = np.array(line[5:len(line)])
    segment_pt = segment.reshape(int(len(segment)/2), 2)
    segment_pt[:,0] = segment_pt[:,0]*width
    segment_pt[:,1] = segment_pt[:,1]*height
    true_segment =  segment_pt.astype(int)


# intersection over union
    true_mask, pred_mask = create_mask(true_segment, pred_segment)
    intersection = true_mask + pred_mask
    intersection[intersection < 2] = 0
    intersection[intersection == 2] = 1
    union = true_mask + pred_mask
    union[union>0] = 1
    int_un = np.sum(intersection)/np.sum(union)
    print(int_un)