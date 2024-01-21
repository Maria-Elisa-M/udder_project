# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 11:12:24 2023

@author: marie
"""
import os
import pandas as pd
from ultralytics import YOLO
import numpy as np
from PIL import Image
import math

# model
model_path = r'runs\pose\train\weights\best.pt'
model = YOLO(model_path)

# test images
dirpath = os.getcwd()
test_labels = os.path.join(dirpath, r'dataset\labels\test')
test_images = os.path.join(dirpath, r'dataset\images\test')

# list images
image_list = [file.replace(".tif", "") for file in os.listdir(test_images)]
#%%
def create_true_points(labelpath):
    global w, h
    with open(labelpath, 'r') as f:
        line = f.readline().strip() 
        points = [float(num) for num in line.split(' ')]
    points_pts = np.array(points[5:])
    points_pts = points_pts.reshape(int(len(points_pts)/3), 3)
    points_pts[:,0] = points_pts[:,0]*w
    points_pts[:,1] = points_pts[:,1]*h
    return points_pts

def euclidean_distance(pointA, pointB):
    pointA = list(pointA)
    pointB = list(pointB)
    distx = (pointA[0]- pointB[0])**2
    disty = (pointA[1]- pointB[1])**2
    return math.sqrt(distx + disty)

def create_true_bbox(labelpath):
    global w, h
    with open(labelpath, 'r') as f:
        line = f.readline().strip() 
        bbox = [float(num) for num in line.split(' ')]
    bbox_pts = np.array(bbox[1:5])
    bbox_w = bbox_pts[2]*w
    bbox_h = bbox_pts[2]*h
    diagonal = math.sqrt(bbox_w**2 +bbox_h**2)
    return diagonal

def euclidean_distance_n(pointA, pointB, diagonal):
    pointA = list(pointA)
    pointB = list(pointB)
    distx = (pointA[0]- pointB[0])**2
    disty = (pointA[1]- pointB[1])**2
    return math.sqrt(distx + disty)/diagonal
#%%
pred_df = pd.DataFrame({"filename": image_list,\
                        "true_LFx": float("nan"),\
                        "true_LFy": float("nan"),\
                        "true_RFx": float("nan"),\
                        "true_RFy": float("nan"),\
                        "true_LRx": float("nan"),\
                        "true_LRy": float("nan"),\
                        "true_RRx": float("nan"),\
                        "true_RRy": float("nan"),\
                        "pred_LFx": float("nan"),\
                        "pred_LFy": float("nan"),\
                        "pred_RFx": float("nan"),\
                        "pred_RFy": float("nan"),\
                        "pred_LRx": float("nan"),\
                        "pred_LRy": float("nan"),\
                        "pred_RRx": float("nan"),\
                        "pred_RRy": float("nan"),\
                        "ed_lf": float("nan"),\
                        "ed_rf": float("nan"),\
                        "ed_lr": float("nan"),\
                        "ed_rr": float("nan"),\
                        "edn_lf": float("nan"),\
                        "edn_rf": float("nan"),\
                        "edn_lr": float("nan"),\
                        "edn_rr": float("nan")})
# for image in tests
for file in image_list:
    file_name = file + ".tif"
    lbl_name = file + ".txt"
    filepath = os.path.join(test_images, file_name)
    labelpath = os.path.join(test_labels, lbl_name)
    
    img = Image.open(filepath)
    w, h = img.size
    results = model(filepath)
    
    true_pts = create_true_points(labelpath)
    pred_pts = results[0].keypoints.xy[0]
    diagonal = create_true_bbox(labelpath)
    
    pred_df.loc[pred_df.filename == file,["true_LFx","true_RFx", "true_LRx", "true_RRx" ]] = true_pts[:, 0]
    pred_df.loc[pred_df.filename == file,["true_LFy","true_RFy", "true_LRy", "true_RRy" ]] = true_pts[:, 1]
    
    pred_df.loc[pred_df.filename == file,["pred_LFx","pred_RFx", "pred_LRx", "pred_RRx" ]] = true_pts[:, 0]
    pred_df.loc[pred_df.filename == file,["pred_LFy","pred_RFy", "pred_LRy", "pred_RRy" ]] = true_pts[:, 1]
    
    pred_df.loc[pred_df.filename == file,"ed_lf"] = euclidean_distance(true_pts[0, :], pred_pts[0, :])
    pred_df.loc[pred_df.filename == file,"ed_rf"] = euclidean_distance(true_pts[1, :], pred_pts[1, :])
    pred_df.loc[pred_df.filename == file,"ed_lr"] = euclidean_distance(true_pts[2, :], pred_pts[2, :])
    pred_df.loc[pred_df.filename == file,"ed_rr"] = euclidean_distance(true_pts[3, :], pred_pts[3, :])
    
    pred_df.loc[pred_df.filename == file,"edn_lf"] = euclidean_distance_n(true_pts[0, :], pred_pts[0, :], diagonal)
    pred_df.loc[pred_df.filename == file,"edn_rf"] = euclidean_distance_n(true_pts[1, :], pred_pts[1, :], diagonal)
    pred_df.loc[pred_df.filename == file,"edn_lr"] = euclidean_distance_n(true_pts[2, :], pred_pts[2, :], diagonal)
    pred_df.loc[pred_df.filename == file,"edn_rr"] = euclidean_distance_n(true_pts[3, :], pred_pts[3, :], diagonal)
    
#%%
pred_df2 = pred_df[["ed_lf","ed_rf","ed_lr", "ed_rr", "edn_lf","edn_rf","edn_lr", "edn_rr"]].agg(["mean", "min", "max", "median"]).reset_index().rename(columns = {"index": "metrics"})


pred_df2.to_csv("keypoint_test_distance.csv", index = False) 
pred_df.to_csv("keypoint_test_predictions.csv", index = False)     
    
    