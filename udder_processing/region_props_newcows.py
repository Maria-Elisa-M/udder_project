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
#%%
feature_list =  ["cow", "filename", "udder_ecc", "udder_circ", "lf_ecc", "rf_ecc", "lb_ecc", "rb_ecc", "lf_circ", "rf_circ", "lb_circ", "rb_circ"]
results_df = pd.DataFrame(columns = feature_list)

def prop_circularity(area, perimeter):
    r = perimeter/(2*np.pi) + 0.5
    circularity = (4*np.pi*area/perimeter*r**2)*(1 - 0.5/r)**2
    return circularity
#%%
cnt = 0
for file in good.filename:
    cow = file.split("_")[0]
    cow_line =dict((key, np.nan) for key in feature_list)
    cow_line["cow"] = cow
    cow_line["filename"] = file
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
    udd_peri = getattr(props[0], 'eccentricity')
    udd_area = getattr(props[0], 'area')
    udd_ecc = getattr(props[0], 'eccentricity')
    udd_circ = prop_circularity(udd_area, udd_peri)
    cow_line["udder_ecc"] = udd_ecc
    cow_line["udder_circ"] = udd_ecc
    # for each segment get region properties
    for key in ws_map.keys():
        val = ws_map[key] 
        quarter_mask = ws_label.copy()
        quarter_mask[quarter_mask != key] = 0
        quarter_mask[quarter_mask != 0 ] = 1
        labels = measure.label(quarter_mask)
        props = measure.regionprops(labels, img)
        qt_ecc = getattr(props[0], 'eccentricity')
        qt_area = getattr(props[0], 'area')
        qt_peri = getattr(props[0], 'perimeter')
        qt_circ = prop_circularity(qt_peri, qt_area)
        cow_line[val+"_ecc"] = qt_ecc
        cow_line[val+"_circ"] = qt_circ
#%%
    cnt +=1
    print(f"{cnt}: {cow}")
    temp = pd.DataFrame(cow_line, index = [0])
    results_df = pd.concat([results_df, temp], axis= 0, ignore_index=True)

results_df.to_csv(os.path.join("udder_features", "region_props_newcows.csv"), index = False)
