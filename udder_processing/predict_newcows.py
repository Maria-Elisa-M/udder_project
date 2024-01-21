# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 09:13:04 2023

@author: marie
"""
import os
from ultralytics import YOLO
import numpy as np
from tifffile import imwrite
import cv2
import pandas as pd

dirpath = os.getcwd()
# video path 
file_list = os.listdir("arrays")

# model path
model_path = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r'udder_models')
modelpath_classify = os.path.join(model_path, r"frame_classify\runs\classify\train\weights\best.pt")
modelpath_segment = os.path.join(model_path, r"udder_segment\runs\segment\train\weights\best.pt")
modelpath_keypoints = os.path.join(model_path, r"teat_keypoints\runs\pose\train\weights\best.pt")

model_classify = YOLO(modelpath_classify)
model_segment = YOLO(modelpath_segment)
model_keypoints = YOLO(modelpath_keypoints)

#%%
def save_segment(filename, polygon):
    outpath = os.path.join(r"pred_labels\segments", filename)
    segment = [str(pt) for p in  polygon for pt in p]
    segment = [str(0)] + segment
    with open(outpath, "w") as f:
        f.write(" ".join(segment))

def save_keypoints(filename, kpoints, bbox):
    outpath = os.path.join(r"pred_labels\keypoints", filename)
    points = [str(pt) for p in  kpoints for pt in p]
    points = [str(0)] + [str(p) for p in bbox] + points
    with open(outpath, "w") as f:
        f.write(" ".join(points))

def save_bbox(filename,  bbox):
    outpath = os.path.join(r"pred_labels\bbox", filename)
    bbox = [str(0)] + [str(p) for p in bbox]
    with open(outpath, "w") as f:
        f.write(" ".join(bbox))
        
def mask_img(poly, img):
    h, w = img.shape
    mask2 = np.zeros([h,w]).astype("int16")
    mask = cv2.fillPoly(mask2, np.array([poly]).astype(np.int32), color=1)
    masked_im = (img*mask).astype("int16")
    return masked_im

def is_not_dup(arr):
    u, c = np.unique(arr, axis=0, return_counts=True)
    return not (c>1).any()

#%%
cows = []
len_good = []
for filename in file_list:        
    good_frames = []
    depth_array = np.load(os.path.join("arrays", filename), mmap_mode="r")
    nframes = depth_array.shape[0]
    cow = filename.split("_")[0]
#%%
    for j in range(0, nframes):
        outname = "_".join(filename.split("_")[:3])+ "_frame_" + str(j) + ".txt"
        img = depth_array[j]
        imwrite("temp_img.tif", depth_array[j])
        results = model_classify("temp_img.tif")
        prob_array = results[0].probs.data.tolist()
        if prob_array[1] > 0.9:
            results = model_segment("temp_img.tif")
            if (len(results) > 0) & (results[0].masks is not None):
                polyn = (results[0].masks[0].xyn[0]).tolist()
                poly = (results[0].masks[0].xy[0]).tolist()
                bbox = (results[0].boxes.xywhn[0]).tolist()
                save_bbox(outname, bbox)
                save_segment(outname, polyn)
                
                masked_im = mask_img(poly, img)
                os.remove("temp_img.tif")
                imwrite("temp_img.tif", masked_im)
                
                results = model_keypoints("temp_img.tif")
                if (len(results[0].keypoints.xyn[0]) > 0):
                    kpoints = results[0].keypoints.xyn[0].tolist()
                    kpoints2 = np.array(kpoints).reshape((4,2))
                    kpoints = np.hstack((kpoints2, [[2]]*4)).tolist()
                    print(kpoints)
                    os.remove("temp_img.tif")
                    if is_not_dup(kpoints2):
                        save_keypoints(outname, kpoints, bbox)
                        good_frames.append(j)
    
    del depth_array
    len_good.append(len(good_frames))
    cows.append(cow)
    good_frames = [str(p) for p in good_frames]
    with open(os.path.join("frames_tosave", "_".join(filename.split("_")[:3])) + ".txt", "w") as f:
        f.write(",".join(good_frames))


cows_df = pd.DataFrame({"cow": cows, "len_goodframes" :len_good})
cows_df.to_csv("cow_good_frames.csv", index = False)
