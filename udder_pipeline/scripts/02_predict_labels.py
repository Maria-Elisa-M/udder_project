# Maria Elisa Montes
# Working version: predict_newcows
# last update: 2025-02-10

import os
from ultralytics import YOLO
import numpy as np
from tifffile import imwrite
import cv2
import pandas as pd
import json

dirpath = os.getcwd()
config_path = os.path.join(dirpath, "udder_config.json")

# Open and read the JSON file
with open(config_path, 'r') as file:
    data = json.load(file)

def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# video path
input_path = data["temp_path"]
array_path = os.path.join(input_path, "arrays")
file_list = os.listdir(array_path)

# output_path 
output_path = data["temp_path"]
label_path = os.path.join(output_path, "pred_labels")
mk_dir(label_path)

label_list = ["bbox", "segments", "keypoints", "frames_to_save"]
for label_name in label_list:
    mk_dir(os.path.join(label_path, label_name))

# model path
model_path = data["model_path"]
modelpath_classify = os.path.join(model_path, r"frame_classify\train\weights\best.pt")
modelpath_segment = os.path.join(model_path, r"udder_segment\train\weights\best.pt")
modelpath_keypoints = os.path.join(model_path, r"teat_keypoints\train\weights\best.pt")

model_classify = YOLO(modelpath_classify)
model_segment = YOLO(modelpath_segment)
model_keypoints = YOLO(modelpath_keypoints)

#%%
def save_segment(filename, polygon, label_path):
    outpath = os.path.join(label_path,"segments", filename)
    segment = [str(pt) for p in  polygon for pt in p]
    segment = [str(0)] + segment
    with open(outpath, "w") as f:
        f.write(" ".join(segment))

def save_keypoints(filename, kpoints, bbox, label_path):
    outpath = os.path.join(label_path, "keypoints", filename)
    points = [str(pt) for p in  kpoints for pt in p]
    points = [str(0)] + [str(p) for p in bbox] + points
    with open(outpath, "w") as f:
        f.write(" ".join(points))

def save_bbox(filename,  bbox, label_path):
    outpath = os.path.join(label_path, "bbox", filename)
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
    depth_array = np.load(os.path.join(array_path, filename), mmap_mode="r")
    nframes = depth_array.shape[0]
    cow = filename.split("_")[0]
#%%
    for j in range(0, nframes):
        outname = filename.replace(".npy", "") + "_frame_" + str(j) + ".txt"
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
                save_bbox(outname, bbox, label_path)
                save_segment(outname, polyn, label_path)
                
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
                        save_keypoints(outname, kpoints, bbox, label_path)
                        good_frames.append(j)
    
    del depth_array
    len_good.append(len(good_frames))
    cows.append(cow)
    good_frames = [str(p) for p in good_frames]
    with open(os.path.join(label_path, "frames_to_save", filename.replace(".npy", ".txt")), "w") as f:
        f.write(",".join(good_frames))


cows_df = pd.DataFrame({"cow": cows, "len_goodframes" :len_good})
cows_df.to_csv(os.path.join(label_path, "cow_good_frames.csv"), index = False)
