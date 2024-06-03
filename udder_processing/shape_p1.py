import os
import watershed_udder as wu
import numpy as np
import pandas as pd
import json
from skimage.measure import label, regionprops, regionprops_table
from skimage import measure

def prop_circularity(area, perimeter):
    circularity = (4*np.pi*area)/(perimeter**2)
    return circularity

dirpath = os.getcwd()
ws_dir = os.path.join("validate_watershed", "watershed_segments")
corr_dir = os.path.join("validate_watershed", "watershed_correspondence")
label_dir = os.path.join(dirpath, "validate_watershed", "pred_labels")
kp_dir = os.path.join(label_dir, "keypoints")
sg_dir = os.path.join(label_dir, "segments")
img_dir = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), "udder_video", "depth_images")
results = pd.read_csv(os.path.join("validate_watershed", "ws_class_predictions_I.csv"))
out_path = os.path.join(dirpath, "features_dict")
good = results[results.thr09 == 1]

cnt = 0
for file in good.filename:
    print(cnt)
    cnt+=1
    shape_dict = {}
    udder = wu.udder_object(file + ".tif", img_dir, label_dir, array = 0)
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
    with open(os.path.join(out_path, "shape", file + ".json"), 'w') as f:
        json.dump(shape_dict, f)