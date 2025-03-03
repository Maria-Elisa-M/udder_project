# Maria Elisa Montes
# Working version: watershed_segments_newcows
# last update: 2025-03-02

import os
from udder_modules import watershed_udder as wu
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table
from skimage import measure
import json


def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok = True)


def prop_circularity(area, perimeter):
    circularity = (4*np.pi*area)/(perimeter**2)
    return circularity

dirpath = os.getcwd()
config_path = os.path.join(dirpath, "udder_config.json")

# Open and read the JSON file
with open(config_path, 'r') as file:
    data = json.load(file)

# label path
input_path = data["temp_path"]
output_path = data["output_path"]
label_dir = os.path.join(input_path, "pred_labels")

kp_dir = os.path.join(label_dir, r"keypoints")
sg_dir = os.path.join(label_dir, r"segments")
ws_dir = os.path.join(label_dir, r"watershed_segments")
corr_dir = os.path.join(label_dir, r"watershed_correspondence")
results = pd.read_csv(os.path.join(label_dir, "ws_class_predictions.csv"))
good = results[results.thr09 == 1]

img_dir = os.path.join(input_path, "depth_images")

out_path_shape = os.path.join(output_path, "features_dict", "shape")
mk_dir(out_path_shape)

cnt = 0
for file in good.filename:
    print(f"{file}: {cnt}")
    cnt+=1
    shape_dict = {}
    udder = wu.udder_object(file, label_dir, array = 0, img_dir = img_dir) 
    img = udder.img
    ws_label = np.load(os.path.join(ws_dir, file + ".npy"))
    kp_ws = pd.read_csv(os.path.join(corr_dir, file +".csv")).loc[0].to_dict()
    new_kp, kp_ws = wu.update_kp(kp_ws, ws_label, img)
    ws_map = dict((v, k) for k, v in kp_ws.items())
    
    udd_mask = udder.get_mask()
    labels = measure.label(udd_mask)
    props = measure.regionprops(labels, img)
    udd_peri = getattr(props[0], 'perimeter')
    udd_area = getattr(props[0], 'area')
    udd_ecc = getattr(props[0], 'eccentricity')
    udd_circ = prop_circularity(udd_area, udd_peri)
    
    shape_dict['udder'] = {'circ': udd_circ, 'exc': udd_ecc, 'peri':udd_peri, 'area': udd_area}
    
    mask0 = np.zeros(udder.size)
    quarters_dict = {}
    for key in kp_ws.keys():
        label = kp_ws[key]
        quarter_mask = mask0.copy()
        rows, cols = np.where(ws_label == label)
        quarter_mask[rows, cols] = 1
        labels = measure.label(quarter_mask)
        props = measure.regionprops(labels, img)
        qt_ecc = getattr(props[0], 'eccentricity')
        qt_area = getattr(props[0], 'area')
        qt_peri = getattr(props[0], 'perimeter')
        qt_circ = prop_circularity(qt_area,qt_peri)
        quarters_dict[key] = {'circ':qt_circ, 'exc':qt_ecc, 'peri':qt_peri, 'area':qt_area}
    
    shape_dict["quarters"] = quarters_dict
    with open(os.path.join(out_path_shape, file + ".json"), 'w') as f:
        json.dump(shape_dict, f)