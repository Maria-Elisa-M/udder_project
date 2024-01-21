import pandas as pd
import os
import numpy as np
from tifffile import imwrite

def mk_dir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

# good frames
dirpath = os.getcwd()
frames_df = pd.read_csv(os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r"udder_labels\frame_class_list.csv"))
good_frames = frames_df[frames_df.frame_class ==1]
good_frames.insert(0, "file",  ["_".join(file.split("_")[:3]) for file in good_frames.filename], True)
image_path = "color_images"
array_path = "arrays"
array_list = []
cow_dirs = [f.name for f in os.scandir(array_path ) if  f.is_dir()]
for cow in cow_dirs:
    cow_path = os.path.join(array_path , cow)
    files =  [file for file in os.listdir(cow_path) if "color" in file]
    array_list.extend(files)
    
for array in array_list:
    cow = array.split("_")[0]
    filename = "_".join(array.split("_")[:3])
    src = os.path.join(array_path, cow, array)
    depth_array = np.load(src, mmap_mode="r")
    cow_gf = good_frames[good_frames.file == filename]["filename"]
    for frame in cow_gf:
        frame_name = frame + ".png"
        j = int(frame.split("_")[-1])-1
        out_path = os.path.join(image_path, cow)
        mk_dir(out_path)
        imwrite(os.path.join(out_path, frame_name), depth_array[j])
    print(f"{cow} ----------- done!")