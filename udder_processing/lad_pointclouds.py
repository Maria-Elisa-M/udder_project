import os
import watershed_udder as wu
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

import pyrealsense2 as rs
from gurobi_optimods.regression import LADRegression
import plotly.graph_objects as go
from astropy.convolution import Gaussian2DKernel, convolve,interpolate_replace_nans

def points_toworld(points):
    points2 = points.copy()
    for i in range(len(points)):
        points2[i, :] = rs.rs2_deproject_pixel_to_point(intr, [points[i, 0], points[i, 1]], points[i, 2])
    return points2
    
# get image points
def pts_fromimg(coords, img):
    values = img[coords[:,1],  coords[:,0]]
    idx = ~ np.isnan(values)
    pts = np.column_stack((np.transpose(coords[idx,0]), np.transpose(coords[idx,1]), np.transpose(values[idx]))).astype(float)
    return pts
# 
def ols_plane(p, n):
    plane_pts = p[p[:, 2].argsort()]
    A = np.column_stack((np.ones((n,1)), plane_pts[:n, :2]))
    B = np.reshape(plane_pts[:n, 2], (n,1))
    plane = np.linalg.inv(np.transpose(A)@A)@np.transpose(A)@B
    return plane

def lad_plane(p):
    # Split data for fit assessment
    x = p[:, :2]
    y = p[:, 2]
    # Fit model and obtain predictions
    lad = LADRegression()
    lad.fit(x, y)
    return lad

def tilt_ponts(plane, old_points):
    a = plane[1]
    b = plane[2]
    c = -1
    d = plane[0]
    new_z = a *old_points[:, 0]+ b * old_points[:,1] + d
    new_points = old_points.copy()
    new_points[:, 2] = new_points[:, 2] - new_z
    return new_points
# 
def down_sample(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downpcd = pcd.voxel_down_sample(voxel_size=0.005)
    return np.asarray(downpcd.points)



# list files 
dirpath = os.getcwd()
label_dir = os.path.join(dirpath, "pred_labels")
ws_dir = os.path.join(label_dir,"watershed_segments")
corr_dir = os.path.join(label_dir, "watershed_correspondence")
kp_dir = os.path.join(label_dir, "keypoints")
sg_dir = os.path.join(label_dir, "segments")
img_dir = os.path.join("depth_images")
results = pd.read_csv(os.path.join(label_dir, "ws_class_predictions_II.csv"))
out_path = os.path.join(dirpath, "point_clouds")




video_path =  os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), "udder_video", "video_files", "example_video.bag")
good = results[results.thr09 == 1]

# get camera parameters
config = rs.config()
rs.config.enable_device_from_file(config, video_path, repeat_playback = False)
pipeline = rs.pipeline()
cfg = pipeline.start(config) # Start pipeline and get the configuration it found
profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
scale = 0.001
gk = Gaussian2DKernel(x_stddev=1)

teat_dict  = {"lf": 1, "rf":2, "lb":3, "rb":4}


for file in good.filename:
    udder = wu.udder_object(file + ".tif", img_dir, label_dir, array = 0)
    # read image
    img = udder.img
    imgr = udder.img.copy().astype(float)
    imgr[imgr ==0] = np.nan
    img = interpolate_replace_nans(imgr, gk)
    
    # read labels
    segment = udder.get_segment()
    kppoints = udder.get_keypoints()
    # reas WS segmentation
    ws_label = np.load(os.path.join(ws_dir, file + ".npy"))
    kp_ws = pd.read_csv(os.path.join(corr_dir, file +".csv")).loc[0].to_dict()
    # ws_map = dict((v, k) for k, v in kp_ws.items())
    # ws_map[0] = "bg"
    new_kp, kp_ws = wu.update_kp(kp_ws, ws_label, img)
    
    kp_locs = np.array([np.array(value) for value in new_kp.values()])
    kp_points = pts_fromimg(kp_locs, img)
    kp_points2 = kp_points.copy()
    kp_points2[:, 2] = kp_points[:, 2] *scale
    kp = points_toworld(kp_points2)
    plane = ols_plane(kp, 3)
    kp2 = tilt_ponts(plane, kp)
    # kp_dict = {key: kp2[i, :].tolist() for i, key in enumerate(new_kp.keys())}
    kp_dict = {key: {} for key in new_kp.keys()}
    
    mask = udder.get_mask()
    rows, cols = np.where(mask == 1)
    ud_locs = np.column_stack([np.transpose(cols), np.transpose(rows)])
    ud_points = pts_fromimg(ud_locs, img)
    ud_points2 = ud_points.copy()
    ud_points2[:, 2] = ud_points2[:, 2] *scale
    ud = points_toworld(ud_points2)
    ud_t = tilt_ponts(plane, ud)
    r = ud_points[:, 1].astype(int)
    c = ud_points[:, 0].astype(int)
    ws_array = ws_label[r, c]

    kp_array = np.zeros((ud.shape[0], 1))
    for key in new_kp.keys():
        y = new_kp[key][1]
        x = new_kp[key][0]
        kidx = np.where((ud_points[:, 0] == x) & (ud_points[:, 1] == y))[0]
        kp_dict[key]["tidx"] = int(kidx[0])
        kp_dict[key]["xyz"] = ud[kidx, :][0].tolist()
        kp_array[kidx] = teat_dict[key]
    
    poly = np.round([[coord[1] * udder.size[0]-1, coord[0]* udder.size[1]-1] for coord in segment]).astype(int)
    sg_locs =  np.column_stack([np.transpose(poly[:, 1]), np.transpose(poly[:, 0])])
    sg_points = pts_fromimg(sg_locs, img)
    sg_points2 = sg_points.copy()
    sg_points2[:, 2] = sg_points2[:, 2] *scale
    sg = points_toworld(sg_points2)
    sg_t = tilt_ponts(plane, sg)
    
    plane = lad_plane(sg_t)
    x_test = ud_t[:, :2]
    lad_pred = plane.predict(x_test)
    
    keep = ud_t[:, 2]<lad_pred
    ud_f = ud_t[keep, :]
    ws_arrayf = ws_array[keep]
    kp_array_tf = kp_array[keep]
    # get quarter points after filtering
    quarter_dict = {}
    for key in kp_ws.keys():
        qidx = np.where(ws_arrayf == kp_ws[key])[0]
        quarter_dict[key] = ud_f[qidx,:].tolist()

    for key in new_kp.keys():
        kidx = np.where(kp_array_tf == teat_dict[key])[0]
        if len(kidx) > 0:
            kp_dict[key]["tidx_tf"] = int(kidx[0])
            kp_dict[key]["xyz_tf"] = ud_f[kidx, :][0].tolist()
        else:
            kp_dict[key]["tidx_tf"] = -1 # use this to filter later
            kp_dict[key]["xyz_tf"] = []
        
    # save outputs
    np.save(os.path.join(out_path, "raw", file + ".npy"), ud)
    np.save(os.path.join(out_path, "udder", file + ".npy"), ud_f)
    
    with open(os.path.join(out_path, "keypoints", file + ".json"), 'w') as f:
        json.dump(kp_dict, f)
    with open(os.path.join(out_path, "quarters", file + ".json"), 'w') as f:
        json.dump(quarter_dict, f)