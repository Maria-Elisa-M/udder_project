import os 
import pandas as pd
import shutil

# class sest df
sets_df = pd.read_csv("frameclass_sets.csv")

# directories
dirpath = os.getcwd()
img_dir = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r'udder_video\depth_images')
old_dir = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r'udder_dcc\images')
data_dir = os.path.join(dirpath, r"frame_classify\data")
# data collection groups
imgdir_dict = {20210625:{"lab": old_dir}, \
              20211022: {"lab": old_dir}, \
              20231117:{"guilherme": img_dir , \
                        "maria": img_dir}}

clas_dict = {0:"bad", 1:"good"}

def mk_dest_dir(data_dir , file_set, file_class):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    set_dir = os.path.join(data_dir, file_set)
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)
    class_dir = os.path.join(set_dir, file_class)
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)

for file in sets_df.filename:
    file_name = file + ".tif"
    file_line = sets_df[sets_df.filename == file]
    # find source directory
    file_date = file_line["date"].values[0]
    computer  = file_line["computer"].values[0]
    src_dir = os.path.join(imgdir_dict[file_date][computer], file_name.split("_")[0])
    img_dir = os.path.join(src_dir, file_name)
    print(img_dir)
    # find/create dest directoy
    file_set = file_line["set_name"].values[0]
    file_class = clas_dict[file_line["frame_class"].values[0]]
    mk_dest_dir(data_dir , file_set, file_class)
    dest_dir = os.path.join(data_dir , file_set, file_class)
    print(dest_dir)
    # copy image
    shutil.copy(img_dir, dest_dir)