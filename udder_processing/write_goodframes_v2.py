import os
import numpy as np
from tifffile import imwrite

# list of files
inpath = "frames_tosave"
array_path = "arrays"
depthpath = "depth_images"
file_list = os.listdir(array_path)

def mk_dir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

# for file in list  read content
for file in file_list:
    cow = file.split("_")[0]
    src = os.path.join(array_path, file)
    depth_array = np.load(src, mmap_mode="r")
    good_frames = "_".join(file.split("_")[:3]) +".txt"
    print(cow)
    with open(os.path.join(inpath, good_frames), "r") as f:
        frames = f.read()
        if frames != "":
            frames = [int(num) for num in frames.split(",")]
            selected_frames = frames
            
            for frame in selected_frames:
                frame_name = "_".join(file.split("_")[:3]) +"_frame_" + str(frame) + ".tif"
                out_path = os.path.join(depthpath, cow)
                mk_dir(out_path)
                imwrite(os.path.join(out_path, frame_name), depth_array[frame])
            print("....done!")