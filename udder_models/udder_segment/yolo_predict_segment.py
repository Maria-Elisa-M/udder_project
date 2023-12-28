# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 13:27:03 2023

@author: marie
"""

import os
import pandas as pd
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

# model
model_path = r'runs\segment\train\weights\best.pt'
model = YOLO(model_path)

# test images
dirpath = os.getcwd()
test_labels = os.path.join(dirpath, r'dataset\labels\test')
test_images = os.path.join(dirpath, r'dataset\images\test')

# list images
image_list = [file.replace(".tif", "") for file in os.listdir(test_images)]

def create_true_mask(labelpath):
    global w, h
    with open(labelpath, 'r') as f:
        line = f.readline().strip() 
        segment = [float(num) for num in line.split(' ')]
    segment_pts = np.array(segment[1:len(segment)])
    segment_pts = segment_pts.reshape(int(len(segment_pts)/2), 2)
    segment_pts[:,0] = segment_pts[:,0]*w
    segment_pts[:,1] = segment_pts[:,1]*h
    mask2 = np.zeros([h,w]).astype("int16")
    mask = cv2.fillPoly(mask2, np.array([segment_pts]).astype(np.int32), color=1)
    return mask

def create_pred_mask(segment):
    global w, h
    mask2 = np.zeros([h,w]).astype("int16")
    mask = cv2.fillPoly(mask2, np.array([segment]).astype(np.int32), color=1)
    return mask
#%%
pred_df = pd.DataFrame({"filename": image_list,\
                        "intersection_union": float("nan")})
# for image in tests
for file in image_list:
    file_name = file + ".tif"
    lbl_name = file + ".txt"
    filepath = os.path.join(test_images, file_name)
    labelpath = os.path.join(test_labels, lbl_name)
    
    img = Image.open(filepath)
    w, h = img.size
    results = model(filepath)
    pred_segment = results[0].masks[0].xy[0]
    
    # true mask
    true_mask = create_true_mask(labelpath)
    # pred mask
    pred_mask = create_pred_mask(pred_segment)
    
    # intersection
    true_pred = true_mask + pred_mask
    intersection = len(true_pred[true_pred > 1])
    union = len(true_pred[true_pred > 0])
    
    inte_union = intersection/union
    pred_df.loc[pred_df.filename == file, "intersection_union"] = inte_union

#%%
pred_df2 = pred_df["intersection_union"].agg(["mean", "min", "max", "median"]).reset_index().rename(columns = {"index": "metrics"})

    
pred_df2.to_csv("segment_test_intersection_union.csv", index = False) 
pred_df.to_csv("segment_test_predictions.csv", index = False)  
    
