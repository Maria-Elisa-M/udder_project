# Maria Elisa Montes
# Working version: watershed_segments_newcows
# last update: 2025-03-02

import os
import numpy as np
from tifffile import imwrite
import json
import pandas as pd

def mk_dir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

dirpath = os.getcwd()
config_path = os.path.join(dirpath, "udder_config.json")

# Open and read the JSON file
with open(config_path, 'r') as file:
    data = json.load(file)

# label path
input_path = data["temp_path"]
output_path = data["temp_path"]
label_dir = os.path.join(input_path, "pred_labels")
results = pd.read_csv(os.path.join(label_dir, "ws_class_predictions.csv"))
good = results[results.thr09 == 1]

# list of files
array_path = os.path.join(output_path,"arrays")
depthpath = os.path.join(output_path, "depth_images")
mk_dir(depthpath)

for file in good.filename:
    print(file)
    cow = file.split("_")[0]
    frame = int(file.split("_")[-1])
    array_name = "_".join(file.split("_")[:-2]) + ".npy"
    frame_name= file + ".tif"
    src = os.path.join(array_path, array_name)
    depth_array = np.load(src, mmap_mode="r")
    imwrite(os.path.join(depthpath, frame_name), depth_array[frame])