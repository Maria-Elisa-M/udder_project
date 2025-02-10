# Maria Elisa Montes
# Working version: watershed_segments_newcows
# last update: 2025-02-10

import numpy as np
import os
from udder_modules import watershed_udder as wu
import pandas as pd
import json

dirpath = os.getcwd()
config_path = os.path.join(dirpath, "udder_config.json")

# Open and read the JSON file
with open(config_path, 'r') as file:
    data = json.load(file)

def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# label path
input_path = data["temp_path"]
label_dir = os.path.join(input_path, "pred_labels")

kp_dir = os.path.join(label_dir, r"keypoints")
sg_dir = os.path.join(label_dir, r"segments")
out_dir = os.path.join(label_dir, r"watershed_segments")
out_dir2 = os.path.join(label_dir, r"watershed_correspondence")

mk_dir(out_dir)
mk_dir(out_dir2)

# list of files
good_frame_path =  os.path.join(label_dir, "frames_to_save")
array_path = os.path.join(input_path, "arrays")
file_list = os.listdir(array_path)

# for file in list  read content
for file in file_list:
    cow = file.split("_")[0]
    src = os.path.join(array_path, file)
    depth_array = np.load(src, mmap_mode="r")
    good_frames = file.replace(".npy", ".txt")
    # print(file)
    with open(os.path.join(good_frame_path, good_frames), "r") as f:
        frames = f.read()
        if frames != "":
            frames = [int(num) for num in frames.split(",")]
            print(f"\n{cow}: {len(frames)}")
            cnt = 1
            # print(frames)
            for frame in frames:
                filename = file.replace(".npy", "") +"_frame_" + str(frame)
                # print(filename)
                img = depth_array[frame]
                udder = wu.udder_object(filename, label_dir, array = img) # no im_dir bacause it is from array
                udder_shp = udder.get_shape()
                udder_box = udder.get_box()
                points = udder.get_keypoints()
                udder_box = udder.get_keypoints()
                udder_mask = udder.get_mask()
                masked_udder = udder.img*udder_mask
                mask1 = np.zeros(udder.size)
                points2 =np.round(points,0).astype(int)

                lf_kp = points[0, :2]
                rf_kp = points[1, :2]
                lb_kp = points[2, :2]
                rb_kp = points[3, :2]

                new_front = wu.sep_points(rf_kp, lf_kp, udder_shp, udder_box)
                points2[0, :2] = new_front[0]
                points2[1, :2] = new_front[1]

                new_back = wu.sep_points(rb_kp, lb_kp, udder_shp, udder_box)
                points2[2, :2] = new_back[0]
                points2[3, :2] = new_back[1]

                labels = wu.watershed_labels(points2, udder)
                np.save(os.path.join(out_dir, file + ".npy"), labels)
                
                temp = pd.DataFrame(wu.find_correspondence(points2, labels), index = [0])
                temp.to_csv(os.path.join(out_dir2, file.replace(".tif", ".csv")), index = False)
                
                print(f"{cnt}: {file}")
                cnt +=1