# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:27:34 2024

@author: marie
"""

import os
import watershed_udder as wu
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyrealsense2 as rs
from astropy.convolution import Gaussian2DKernel, convolve,interpolate_replace_nans
import open3d as o3d
from scipy.ndimage import gaussian_filter
from scipy.linalg import lstsq
from scipy.spatial import Delaunay
import math
import functools 

#%%
import vedo
import napari
import numpy as np
from pygeodesic import geodesic
import napari_process_points_and_surfaces as nppas

#%%

# list files 
dirpath = os.getcwd()
ws_dir = r"validate_watershed\watershed_segments"
corr_dir = r"validate_watershed\watershed_correspondence"
label_dir = os.path.join(dirpath, r"validate_watershed\pred_labels")
kp_dir = os.path.join(label_dir, r"keypoints")
sg_dir = os.path.join(label_dir, r"segments")
img_dir = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r"udder_video\depth_images")
results = pd.read_csv(r"validate_watershed\ws_class_predictions_I.csv")
good = results[results.thr09 == 1]
filenames = [file.replace(".npy", "") for file in os.listdir(ws_dir)]
cows = set()
filenames2 = []
for file in filenames: 
    cow = file.split("_")[0]
    if cow not in cows:
        cows.add(cow)
        filenames2.append(file)

video_path =  os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r"udder_video\video_files\example_video.bag")

results_df = pd.DataFrame(columns = ["cow", "filename", "volume", "lf_vol", "rf_vol", "lb_vol", "rb_vol"])

config = rs.config()
rs.config.enable_device_from_file(config, video_path, repeat_playback = False)
pipeline = rs.pipeline()
cfg = pipeline.start(config) # Start pipeline and get the configuration it found
profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
# depth_sensor = profile.as_video_stream_profile().get_device().first_depth_sensor()
# depth_scale = depth_sensor.get_depth_scale()

# quarter assignment - according to ws_map 
# lf - yelow (255,255,0)
# rf - cian (0, 255, 255)
# lb - magenta (255, 0, 255)
# rb - white (255,255,255)
# background - black
color_dict = {"lf":[1,1,0], "rf": [0, 1, 1], "lb":[1, 0,1], "rb":[0.5,0.5,0.5], "bg": [0, 0, 0]}

for file in good.filename[:1]:
    cow = file.split("_")[0]
    cow_line = {"cow": cow, "filename":file, "volume": np.nan, "lf_vol":np.nan, "rf_vol": np.nan, "lb_vol":np.nan, "rb_vol":np.nan}
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
    ws_map[0] = "bg"
    new_kp = wu.update_kp(kp_ws, ws_label, img)