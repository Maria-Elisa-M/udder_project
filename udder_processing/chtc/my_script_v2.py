import os 
import pandas as pd
import numpy as np
import json 
import sys

import point_cloud_utils as pcu
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, Sum, CompoundKernel

def down_sample(points):
    downpcd = pcu.downsample_point_cloud_on_voxel_grid(voxel_size=0.005, points = points)
    return downpcd

dirpath = os.getcwd()
job_num = str(sys.argv[1])

job_path = os.path.join(dirpath, job_num)
pcd_path = os.path.join(job_path, "point_clouds")
kp_path = os.path.join(pcd_path, "keypoints")
quarter_path = os.path.join(pcd_path, "quarters")

out_path = os.path.join(job_path, "features_dict")

filenames = [file.replace(".json", "") for file in os.listdir(kp_path) if file.endswith(".json")]
kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e2))  + WhiteKernel()

skip_files = []
cnt = 0
for file in filenames:
    print(cnt)
    cnt+=1
    #start_time = time.time()
    
    with open(os.path.join(quarter_path, file + ".json")) as f:
        quarter_dict = json.load(f)
    
    with open(os.path.join(kp_path, file + ".json")) as f:
        kp_dict = json.load(f)
    
    #end_time = time.time()
    #print(f"Reding files: {end_time-start_time}\n")

    pcd_dict = {}
    teat_dict = {} 
    
    for key in quarter_dict.keys():
        skip = 0
        quarter = np.array(quarter_dict[key])
        teat = np.array(kp_dict[key]['xyz_tf'])
        dist_teat = np.array([np.linalg.norm(teat[:2]-point[:2]) for point in quarter])
        max_dist_quarter = np.max(dist_teat)
        
        delta_z = []
        radi = []
        z_val = []
        rad = 0.001
        last_rad = 0
        step = 0.002
        last_z = teat[2]
        
        while rad < max_dist_quarter:
            # sort the distances
            # points within 1mm 
            condition = [(d <= rad) & (d > last_rad) for d in dist_teat]
            if sum(condition) >0:
                circle_idx = np.where(condition)[0]
                cirle_dist = dist_teat[circle_idx]
                circle_pts = quarter[circle_idx]
                circle_z = np.min(circle_pts[:, 2])
                
                # max distance in circle
                max_dist_circle = np.max(cirle_dist)
                
                # look at the z change
                dif = abs(circle_z - last_z)
                delta_z.append(dif)
                radi.append(rad)
                z_val.append(circle_z)
                
                # update stuff
                last_rad = rad   
                last_z = circle_z
            rad += step 
        delta_z = np.array(delta_z)
        radi = np.array(radi)
        z_val = np.array(z_val)
        
        small_gradient = [(d >= 0) & (d < 0.001) for d in delta_z]
        small_idx = np.where(small_gradient)[0]
        candidates = np.array(radi[small_idx])
        candidates_h = np.array(z_val[small_idx])
        
        try:
            cand_idx = np.where((candidates[1:] - candidates[:-1]) > step*2)[0][0]
            teat_radius = candidates[cand_idx+1] -step
            teat_z = candidates_h[cand_idx+1]
        except:
            skip = 1
            skip_files.append(file)
    
        if skip == 0:
            # flag the points that are teat 
            base_idx = np.where(dist_teat < teat_radius)[0]
            lowpts_idx = np.where(quarter[:, 2] < teat_z)[0]
            base_coords = np.row_stack((quarter[base_idx], quarter[lowpts_idx]))
            quarter2 = quarter.copy()
            quarter2[base_idx, 2] = np.nan
            quarter2[lowpts_idx, 2] = np.nan
            
            dspc = down_sample(quarter2)
            
            missing = dspc[np.isnan(dspc[:, 2]), :2]
            observed = dspc[~np.isnan(dspc[:, 2]), :]
            
            if len(observed) > 50:
                # start_time = time.time()
                gaussian_process = GaussianProcessRegressor(kernel=kernel)
                gaussian_process.fit(observed[:,:2], observed[:,2])
                vals = gaussian_process.predict(missing[:, :2], return_std=False)
                predicted = np.column_stack([missing, np.transpose(vals)])
                pcd_dict[key] = {"teat_pts": base_coords.tolist(), "obs_pts": observed.tolist(), "pred_pts": predicted.tolist()}
                # end_time = time.time()
                # print(f"Running model: {end_time-start_time}\n")

                leng_all = np.array([np.linalg.norm(teat - point) for point in predicted])
                teat_base = predicted[np.argsort(leng_all)[0]]
                teat_length = leng_all[np.argsort(leng_all)[0]]
                
                teat_dict[key] = {"bottom": teat.tolist(), "tip": teat_base.tolist(), "length": teat_length}
        
    with open(os.path.join(pcd_path, "teat", file + ".json"), 'w') as f:
        json.dump(pcd_dict, f)

    with open(os.path.join(out_path, file + ".json"), 'w') as f:
        json.dump(teat_dict, f)

with open(os.path.join(job_path, job_num +"_skip_files.txt"), 'w') as f:
    for line in skip_files:
            f.write(f"{line}\n") 