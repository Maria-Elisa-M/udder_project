# Maria Elisa Montes
# Working version: watershed_segments_newcows
# last update: 2025-03-02

import numpy as np 
import pandas as pd
import os
import json
import vedo
import napari
import open3d as o3d
from scipy.spatial import Delaunay
from pygeodesic import geodesic
import napari_process_points_and_surfaces as nppas

def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok = True)

def teat_distance(teat_names, A_idx, B_idx, udder_points):
    A = udder_points[A_idx]
    B = udder_points[B_idx]
    distance, path1 = geoalg.geodesicDistance(A_idx, B_idx)
    euc_dist = np.sqrt((A[0]-B[0])**2 +(A[1]-B[1])**2 + (A[1]-B[1])**2)
    result = {"geo":distance, "path": path1.tolist(), "eu":euc_dist}
    return result

def angle_area(p, p1, p2):
    v = p1 - p
    u = p2 - p
    vnorm = np.linalg.norm(v)
    unorm = np.linalg.norm(u)
    uv = np.dot(u,v)
    costh = uv/(vnorm*unorm)
    angle_rad = np.arccos(costh)
    angle = np.rad2deg(angle_rad)
    area = (np.sin(angle_rad)*unorm*vnorm)/2
    result = {"angle": angle, "area": area, "loc":p.tolist()}
    return result

def pcd_tomesh(points):
    tri = Delaunay(np.array(points[:, :2]))
    surface = o3d.geometry.TriangleMesh()
    surface.vertices = o3d.utility.Vector3dVector(points)
    surface.triangles = o3d.utility.Vector3iVector(tri.simplices)
    mesh = vedo.Mesh([np.array(surface.vertices), np.array(surface.triangles)])
    return mesh

edges_dict = {"front" :["rf", "lf"], "back" : ["rb", "lb"], "right" :["rf", "rb"], "left" : ["lf", "lb"]}
nodes_dict = {"lf": ["rf", "lb"] ,"rf": ["lf", "rb"], "lb": ["lf", "rb"], "rb":["rf", "lb"]}

# paths   
dirpath = os.getcwd()
config_path = os.path.join(dirpath, "udder_config.json")

# Open and read the JSON file
with open(config_path, 'r') as file:
    data = json.load(file)

# paths
# label path
input_path = data["temp_path"]
output_path = data["output_path"]
label_dir = os.path.join(input_path, "pred_labels")
pcd_path = os.path.join(input_path, "point_clouds")
udder_path = os.path.join(pcd_path, "udder")
raw_path = os.path.join(pcd_path, "raw")
quarter_path = os.path.join(pcd_path, "quarters")
kp_path = os.path.join(pcd_path, "keypoints")

# creaater folders to save features
out_path = os.path.join(output_path, "features_dict")
mk_dir(out_path)
feature_list = ["volumes", "distance", "angles"]
for folder in feature_list:
    mk_dir(os.path.join(out_path, folder))

filenames = [file.replace(".json", "") for file in os.listdir(kp_path)]

# read udder and quarter point coulds
count =0
for file in filenames:
    print(count)
    count +=1
    udder_pc = np.load(os.path.join(udder_path, file + ".npy"))
    
    with open(os.path.join(quarter_path, file + ".json")) as f:
        quarter_dict = json.load(f)
    
    with open(os.path.join(kp_path, file + ".json")) as f:
        kp_dict = json.load(f)
    
    volumes_dict ={}
    distance_dict = {}
    
    ud_mesh = pcd_tomesh(udder_pc)
    volume = ud_mesh.volume()*1000
    sarea = ud_mesh.area()
    
    volumes_dict["udder"] ={"volume": volume, "sarea":sarea}
    geoalg = geodesic.PyGeodesicAlgorithmExact(ud_mesh.points(), ud_mesh.cells)
    
    for side in edges_dict.keys():
        teat_names = edges_dict[side]
        A_idx = kp_dict[teat_names[0]]['tidx_tf']
        B_idx = kp_dict[teat_names[1]]['tidx_tf']
        if((A_idx != -1) and (B_idx != -1)):
            dist = teat_distance(teat_names, A_idx, B_idx, udder_pc)
            distance_dict[side] = dist 
    
    qvolumes_dict = {}
    for key in quarter_dict.keys():
        qpoints = np.array(quarter_dict[key])
        q_mesh = pcd_tomesh(qpoints)
        v = q_mesh.volume()*1000
        sa = q_mesh.area()
        qvolumes_dict[key] = {"volume":v, "sarea": sa}
    
    volumes_dict["quarters"] = qvolumes_dict
    
    angle_dict = {}
    for teat in nodes_dict.keys():
        p = np.array(kp_dict[teat]['xyz'])
        p1 = np.array(kp_dict[nodes_dict[teat][0]]['xyz'])
        p2 = np.array(kp_dict[nodes_dict[teat][1]]['xyz'])
        if ((len(p)>0) and (len(p1)>0) and (len(p2)>0)):
            angle_dict[teat] = angle_area(p, p1, p2)
    
    with open(os.path.join(out_path, "volumes", file + ".json"), 'w') as f:
        json.dump(volumes_dict, f)
    with open(os.path.join(out_path, "distance", file + ".json"), 'w') as f:
        json.dump(distance_dict, f)
    with open(os.path.join(out_path, "angles", file + ".json"), 'w') as f:
        json.dump(angle_dict, f)

