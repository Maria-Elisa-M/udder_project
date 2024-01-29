import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import os
import watershed_udder as wu
import pandas as pd

dirpath = os.getcwd()
label_dir = os.path.join(dirpath, r"validate_watershed\pred_labels")
kp_dir = os.path.join(label_dir, r"keypoints")
sg_dir = os.path.join(label_dir, r"segments")
im_dir = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r"udder_video\depth_images")
out_dir = r"validate_watershed\watershed_segments"
out_dir2 = r"validate_watershed\watershed_correspondence"
# filenames = [file.replace(".txt", ".tif") for file in os.listdir(kp_dir)]
df = pd.read_csv(r"validate_watershed\survey_groups.csv")
filenames = [file + ".tif" for file in df.filename if "1003" in file]

cnt = 0
for file in filenames:
    out_name = file.replace(".tif", ".npy")
    udder = wu.udder_object(file,im_dir, label_dir)
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
    np.save(os.path.join(out_dir, out_name), labels)
    
    temp = pd.DataFrame(wu.find_correspondence(points2, labels), index = [0])
    temp.to_csv(os.path.join(out_dir2, file.replace(".tif", ".csv")), index = False)
    
    print(f"{cnt}: {file}")
    cnt +=1