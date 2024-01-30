# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:58:21 2024

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

from scipy.linalg import lstsq
from scipy.spatial import Delaunay
import math
import functools 
import shapely
from shapely import LineString, MultiPoint, Polygon

def points_toworld(points):
    points2 = points.copy()
    for i in range(len(points)):
        points2[i, :] = rs.rs2_deproject_pixel_to_point(intr, [points[i, 0], points[i, 1]], points[i, 2])
    return points2

def rotate_segment(right_kp, left_kp, segment, angle, center):
    k = wu.get_orientation(right_kp, left_kp)
    points = segment
    points2 = points.copy()
    rot_mat = np.array([[np.cos(-k*angle), -np.sin(-k*angle)], [np.sin(-k*angle), np.cos(-k*angle)]])
    #
    points2[:, 0] = points[:, 0] - center[0]
    points2[:, 1] = points[:, 1] - center[1]
    # 
    points2 = np.transpose(np.dot(rot_mat, np.transpose(points2[:, :2])))
    points2[:, 0] = points2[:, 0] + center[0]
    points2[:, 1] = points2[:, 1] + center[1]
    rotated_points = points2.copy()
    return rotated_points

# list files 
dirpath = os.getcwd()
ws_dir = r"validate_watershed\watershed_segments"
corr_dir = r"validate_watershed\watershed_correspondence"
label_dir = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r"udder_labels\labels")
kp_dir = os.path.join(label_dir, r"keypoints")
sg_dir = os.path.join(label_dir, r"segments")
img_dir = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r"udder_video\depth_images")
filenames = [file.replace(".txt", "") for file in os.listdir(kp_dir)]

video_path =  os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r"udder_video\video_files\example_video.bag")
#%%
config = rs.config()
rs.config.enable_device_from_file(config, video_path, repeat_playback = False)
pipeline = rs.pipeline()
cfg = pipeline.start(config) # Start pipeline and get the configuration it found
profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
# depth_sensor = profile.as_video_stream_profile().get_device().first_depth_sensor()
# depth_scale = depth_sensor.get_depth_scale()

#%%
for file in filenames:
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
    new_kp = wu.update_kp(kp_ws, ws_label, img)
    

    try:
        center = wu.get_center(new_kp["rf"], new_kp["lf"])
        angle = wu.get_angle(new_kp["rf"], new_kp["lf"])
        # rotate segment 
        segment = np.array([[coord[0] * udder.size[1], coord[1]* udder.size[0]] for coord in udder.get_segment()])
        # draw line get intersection
        rotated_segment = rotate_segment(new_kp["rf"], new_kp["lf"], segment, angle, center)
        line = LineString([center, (center[0], 0)])
        boundary = Polygon(rotated_segment).boundary
        intersection = shapely.intersection(boundary, line).xy
        udder_rotate = wu.rotate_udder(img, new_kp["rf"], new_kp["lf"])
    except:
        center = wu.get_center(new_kp["rb"], new_kp["lb"])
        angle = wu.get_angle(new_kp["rb"], new_kp["lb"])
        # rotate segment 
        segment = np.array([[coord[0] * udder.size[1], coord[1]* udder.size[0]] for coord in udder.get_segment()])
        # draw line get intersection
        rotated_segment = rotate_segment(new_kp["rb"], new_kp["lb"], segment, angle, center)
        line = LineString([center, (center[0], udder.size[0])])
        boundary = Polygon(rotated_segment).boundary
        intersection = shapely.intersection(boundary, line).xy
        udder_rotate = wu.rotate_udder(img, new_kp["rb"], new_kp["lb"])
        
    plt.imshow(udder_rotate)
    plt.plot(rotated_segment[:,0], rotated_segment[:,1], "*b")
    plt.plot(center[0], center[1], "*b")
    plt.plot(line.xy[0], line.xy[1])
    plt.plot(intersection[0], intersection[1], "*r")
    plt.savefig(os.path.join("temp", file + ".png"))
    plt.close()


#%%
plt.plot(new_kp["lf"][0], new_kp["lf"][1], "*r")
plt.plot(new_kp["rf"][0], new_kp["rf"][1], "*b")
plt.plot(new_kp["lb"][0], new_kp["lb"][1], "*r")
plt.plot(new_kp["rb"][0], new_kp["rb"][1], "*b")
