# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 23:59:21 2024

@author: marie
"""

import os
import watershed_udder as wu
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table
from skimage import measure

# list files 
dirpath = os.getcwd()
label_dir = os.path.join(dirpath, "pred_labels")
ws_dir = os.path.join(label_dir,"watershed_segments")
corr_dir = os.path.join(label_dir, "watershed_correspondence")
kp_dir = os.path.join(label_dir, "keypoints")
sg_dir = os.path.join(label_dir, "segments")
img_dir = os.path.join("depth_images")
results = pd.read_csv(os.path.join(label_dir, "ws_class_predictions_II.csv"))
good = results[results.thr09 == 1]
filenames = [file.replace(".npy", "") for file in os.listdir(ws_dir)]
#%%
results_df = pd.DataFrame(columns = ["cow", "filename", "udder_ecc", "lf_ecc", "rf_ecc", "lb_ecc", "rb_ecc"])
cnt = 0
for file in good.filename:
    cow = file.split("_")[0]
    print(f"{cow}: {cnt}")
    cow_line = {"cow": cow, "filename":file, "udder_ecc": np.nan, "lf_ecc":np.nan, "rf_ecc": np.nan, "lb_ecc":np.nan, "rb_ecc":np.nan}
    # udder object
    udder = wu.udder_object(file + ".tif", img_dir, label_dir, array = 0)
    # read image
    img = udder.img
    # read labels
    segment = udder.get_segment()
    points = udder.get_keypoints()
    # reas WS segmentation
    ws_label = np.load(os.path.join(ws_dir, file + ".npy"))
    kp_ws = pd.read_csv(os.path.join(corr_dir, file +".csv")).loc[0].to_dict()
    ws_map = dict((v, k) for k, v in kp_ws.items())
    
    udd_mask = udder.get_mask()
    labels = measure.label(udd_mask)
    props = measure.regionprops(labels, img)
    udd_ecc = getattr(props[0], 'eccentricity')
    cow_line["udder_ecc"] = udd_ecc
    
    # for each segment get region properties
    for key in ws_map.keys():
        val = ws_map[key] 
        quarter_mask = ws_label.copy()
        quarter_mask[quarter_mask != key] = 0
        quarter_mask[quarter_mask != 0 ] = 1
        labels = measure.label(quarter_mask)
        props = measure.regionprops(labels, img)
        qt_ecc = getattr(props[0], 'eccentricity')
        cow_line[val+"_ecc"] = qt_ecc
#%%
    cnt =+1
    temp = pd.DataFrame(cow_line, index = [0])
    results_df = pd.concat([results_df, temp], axis= 0, ignore_index=True)

results_df.to_csv("region_props_newcows.csv")