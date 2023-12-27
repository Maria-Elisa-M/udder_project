import os 
import pandas as pd
import shutil

# segment sest df
sets_df = pd.read_csv("segment_sets.csv")

# directories
dirpath = os.getcwd()
newimg_dir = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r'udder_video\depth_images')
oldimg_dir = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r'udder_dcc\images')

newlbl_dir = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r'udder_labels\labels\segments')
oldlbl_dir = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r'udder_dcc\labels\segments')

data_imdir = os.path.join(dirpath, r"udder_segment\dataset\images")
data_lbldir = os.path.join(dirpath, r"udder_segment\dataset\labels")

# data collection groups
imgdir_dict = {20210625:{"lab": oldimg_dir}, \
              20211022: {"lab": oldimg_dir}, \
              20231117:{"guilherme": newimg_dir , \
                        "maria": newimg_dir}}
# data collection groups
labeldir_dict = {20210625:{"lab": oldlbl_dir}, \
              20211022: {"lab": oldlbl_dir}, \
              20231117:{"guilherme": newlbl_dir, \
                        "maria":newlbl_dir}}

def mk_dest_dir(data_dir , file_set):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    set_dir = os.path.join(data_dir, file_set)
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)
        
for file in sets_df.filename:
    file_line = sets_df[sets_df.filename == file]
    # find source directory
    file_date = file_line["date"].values[0]
    computer  = file_line["computer"].values[0]
    
    imsrc_dir = imgdir_dict[file_date][computer]
    folder = file.split("_")[0]
    img_dir = os.path.join(imsrc_dir, folder, file + ".tif")
    print(img_dir)
    
    lbsrc_dir = labeldir_dict[file_date][computer]
    lbl_dir = os.path.join(lbsrc_dir, file + ".txt")
    print(lbl_dir)
    
    # find/create dest directoy
    file_set = file_line["set_name"].values[0]
    mk_dest_dir(data_imdir, file_set)
    mk_dest_dir(data_lbldir, file_set)
    imdest_dir = os.path.join(data_imdir, file_set, file + ".tif")
    lbdest_dir = os.path.join(data_lbldir, file_set, file + ".txt")
    print(imdest_dir)
    print(lbdest_dir)
    
    # copy image
    shutil.copy(img_dir, imdest_dir)
    shutil.copy(lbl_dir, lbdest_dir)