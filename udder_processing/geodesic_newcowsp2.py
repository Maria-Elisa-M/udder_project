# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 23:46:04 2024

@author: marie
"""


import vedo
import napari
import numpy as np
from pygeodesic import geodesic
import napari_process_points_and_surfaces as nppas

import os
import watershed_udder as wu
import pandas as pd
import pyrealsense2 as rs
from astropy.convolution import Gaussian2DKernel, convolve,interpolate_replace_nans
import open3d as o3d
from scipy.ndimage import gaussian_filter
from scipy.linalg import lstsq
from scipy.spatial import Delaunay
import math

def points_toworld(points):
    points2 = points.copy()
    for i in range(len(points)):
        points2[i, :] = rs.rs2_deproject_pixel_to_point(intr, [points[i, 0], points[i, 1]], points[i, 2])
    return points2

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

video_path =  os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), "udder_video", "video_files", "example_video.bag")
feature_list = ["cow", "filename", "volume2", "area" \
                "lf_vol2", "rf_vol2", "lb_vol2", "rb_vol2",\
                "lf_area", "rf_area", "lb_area", "rb_area", \
                "front_geo", "back_geo", "front_eu", "back_eu"]

results_df = pd.DataFrame(columns = feature_list)

# get camera parameters
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
teat_dict  = {"lf":1, "rf":2, "lb":3, "rb":4}
front_teats = ["rf", "lf"]
back_teats =  ["rb", "lb"]

cnt = 0
for file in good.filename[5000:10000]:
    cow = file.split("_")[0]
    cow_line = dict((key, np.nan) for key in feature_list)
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
    scale = 0.001
    img = udder.img.copy().astype(float)
    img[img ==0] = np.nan
    kernel = Gaussian2DKernel(x_stddev=1)
    udder_conv = interpolate_replace_nans(img, kernel)
    udder_conv[np.isnan(udder_conv)] = 0
    
    masked_udder =  udder.get_mask() * udder_conv 
    rows, cols = np.nonzero(masked_udder)
    values = masked_udder[rows, cols]
    quarter_lbls = ws_label[rows, cols]
    quarter_colors = quarter_colors = np.array([color_dict[ws_map[point]] if point in ws_map.keys() else [0,0,0]for point in quarter_lbls ])
    udder_points = np.column_stack((np.transpose(cols), np.transpose(rows), np.transpose(values))).astype(float)
    #  for each teat locate the point in the array
    kp_points = np.column_stack((udder_points, np.zeros((len(udder_points),1))))
    for key in new_kp.keys():
        kp_points[(udder_points[:, 0] == new_kp[key][0]) & (udder_points[:, 1] == new_kp[key][1]), 3] = teat_dict[key]
    udder_points[:, 2] = udder_points[:, 2] *scale
    pts = points_toworld(udder_points)

    segment = np.round([[coord[1] * udder.size[0]-1, coord[0]* udder.size[1]-1] for coord in udder.get_segment()]).astype(int)
    cols = segment[:, 1]
    rows = segment[:, 0]
    values = udder_conv[rows, cols]*scale
    segment_points = np.column_stack((np.transpose(cols), np.transpose(rows), np.transpose(values))).astype(float)
    sgpts = points_toworld(segment_points)
    # %%
    filtered = sgpts.copy()
    med = np.mean(sgpts[:,2])
    std = np.std(sgpts[:,2])
    filtered = filtered[sgpts[:,2]<=med+2*std] 
    filtered[:, 2] = gaussian_filter(filtered[:,2], 1, truncate = 2)
    #%%  
    X = np.column_stack((np.ones((len(filtered), 1)), filtered[:, :2]))
    z = np.transpose(filtered[:, 2])
    
    b = np.matrix(z).T
    A = np.matrix(X)
    
    fit, residual, rnk, s = lstsq(A, b)
    
    predz =  fit[1] * pts[:,0] + fit[2] * pts[:,1] + fit[0]
    croped = pts.copy()
    quarter_colors = quarter_colors[croped[:,2] <= predz[:]]
    quarter_lbls = quarter_lbls[croped[:,2] <= predz[:]]
    kp_points = kp_points[croped[:,2] <= predz[:]]
    croped = croped[croped[:,2] <= predz[:]]
    
    plane = croped.copy()
    plane[:, 2] = fit[1] * croped[:,0] + fit[2] * croped[:,1] + fit[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(croped)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
   
    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(plane)
    plane_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    pcd.colors = o3d.utility.Vector3dVector(quarter_colors)
    #o3d.visualization.draw_geometries([pcd, plane_pcd])
    center = plane_pcd.get_center()
    a = fit[1][0]
    b = fit[2][0]
    c = -1
    d = fit[0][0]
    cos_theta = c / math.sqrt(a**2 + b**2 + c**2)
    sin_theta = math.sqrt((a**2+b**2)/(a**2 + b**2 + c**2))
    u_1 = b / math.sqrt(a**2 + b**2 )
    u_2 = -a / math.sqrt(a**2 + b**2)
    rotation_matrix = np.array([[cos_theta + u_1**2 * (1-cos_theta), u_1*u_2*(1-cos_theta), u_2*sin_theta],
                                [u_1*u_2*(1-cos_theta), cos_theta + u_2**2*(1- cos_theta), -u_1*sin_theta],
                                [-u_2*sin_theta, u_1*sin_theta, cos_theta]])
    
    pcd.rotate(rotation_matrix, center = center)
    plane_pcd.rotate(rotation_matrix, center = center)
    xyz = np.asarray(pcd.points)
    floor = np.min(xyz[:, 2])
    xyz[:, 2] = xyz[:, 2]-floor
    xy_catalog = []
    for point in xyz:
        xy_catalog.append([point[0], point[1]])
    tri = Delaunay(np.array(xy_catalog))
    surface = o3d.geometry.TriangleMesh()
    surface.vertices = o3d.utility.Vector3dVector(xyz)
    surface.triangles = o3d.utility.Vector3iVector(tri.simplices)
    
    mesh = vedo.Mesh([np.array(surface.vertices), np.array(surface.triangles)])
    # get udder volume and surface area
    cow_line["volume2"] = mesh.volume()*1000
    cow_line["area"]= mesh.area()

    ud_pts = np.asarray(pcd.points)
    geoalg = geodesic.PyGeodesicAlgorithmExact(mesh.points(), mesh.cells)
    teats = [key for key in new_kp.keys()]
    
    if len(np.intersect1d(back_teats, teats)) == 2:
        A_idxs =  np.where(kp_points[:, 3]== teat_dict[back_teats[0]])
        B_idxs =  np.where(kp_points[:, 3]== teat_dict[back_teats[1]])
        if (len(A_idxs[0]) >0) & (len(B_idxs[0]) >0):
            A_idx = A_idxs[0][0]
            B_idx = B_idxs[0][0]
            A = ud_pts[A_idx]
            B = ud_pts[B_idx]
            distance, path = geoalg.geodesicDistance(A_idx, B_idx)
            euc_dist = np.sqrt((A[0]-B[0])**2 +(A[1]-B[1])**2 + (A[1]-B[1])**2)
            cow_line["back_geo"] = distance
            cow_line["back_eu"] = euc_dist
    
    if len(np.intersect1d(front_teats, teats)) == 2:
        A_idxs =  np.where(kp_points[:, 3]== teat_dict[front_teats[0]])
        B_idxs =  np.where(kp_points[:, 3]== teat_dict[front_teats[1]])
        if (len(A_idxs[0]) >0) & (len(B_idxs[0]) >0):
            A_idx = A_idxs[0][0]
            B_idx = B_idxs[0][0]
            A = ud_pts[A_idx]
            B = ud_pts[B_idx]
            distance, path = geoalg.geodesicDistance(A_idx, B_idx)
            euc_dist = np.sqrt((A[0]-B[0])**2 +(A[1]-B[1])**2 + (A[1]-B[1])**2)
            cow_line["front_geo"] = distance
            cow_line["front_eu"] = euc_dist
    
    map_vals = np.unique(quarter_lbls)
    keys =[k for k in ws_map.keys()]
    map_vals = np.intersect1d(map_vals, np.array(keys))
    for key in map_vals:
        val = ws_map[key]
        if val in kp_ws.keys():
            indices = np.array(np.where(quarter_lbls==key))[0]
            qrt_pts = ud_pts[indices, :]
            qrt_pc = o3d.geometry.PointCloud()
            qrt_pc.points = o3d.utility.Vector3dVector(qrt_pts)
            xyz = np.asarray(qrt_pc.points)
            floor = np.min(xyz[:, 2])
            xyz[:, 2] = xyz[:, 2]- floor
            xy_catalog = []
            for point in xyz:
                xy_catalog.append([point[0], point[1]])
            if len(xy_catalog)>12:
                tri = Delaunay(np.array(xy_catalog))
                surface = o3d.geometry.TriangleMesh()
                surface.vertices = o3d.utility.Vector3dVector(xyz)
                surface.triangles = o3d.utility.Vector3iVector(tri.simplices)
                mesh = vedo.Mesh([np.array(surface.vertices), np.array(surface.triangles)])
                cow_line[val+"_vol2"] = mesh.volume()*1000
                cow_line[val+"_area"] = mesh.area()
    
    cnt +=1
    print(f"{cnt}: {cow}")
    temp = pd.DataFrame(cow_line, index = [0])
    results_df = pd.concat([results_df, temp], axis= 0)

results_df.to_csv(os.path.join("udder_features", "geosedic_newcows_p2.csv"), index = False)