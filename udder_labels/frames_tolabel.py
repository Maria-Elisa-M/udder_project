import os
import pandas as pd
import numpy as np
import shutil

def make_dir(newdir_path):
    try: 
        os.mkdir(newdir_path)
    except:
        pass
    
dirpath = os.getcwd()
tolbl_path = os.path.join(dirpath, r"frames_tolabel")
image_path = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r'udder_video\color_images_test')

class_df = pd.read_csv("frame_class_list.csv")
good_df = class_df[class_df.frame_class == 1]
cow_fileno = good_df[["cow", "filename"]].groupby("cow").agg("count").reset_index()
selframes_df = pd.DataFrame(columns = ["cow", "filename"])
frame_dict = {}

for cow in cow_fileno.cow:
    cow_frames = list(good_df[good_df.cow == cow]["filename"])
    np.random.seed(5)
    np.random.shuffle(cow_frames)
    selected_frames = cow_frames[:30]
    temp_df = pd.DataFrame({"cow": [cow]*30, "filename": selected_frames} )
    selframes_df = pd.concat([selframes_df, temp_df], axis = 0, ignore_index = True)
    frame_dict[cow] = selected_frames
    
selframes_df.to_csv("frames_tolabel.csv", index = False)

for cow in list(frame_dict.keys())[:1]:
    src_path = os.path.join(image_path, str(cow))
    dest_path = os.path.join(tolbl_path, str(cow))
    make_dir(dest_path)
    cow_frames = frame_dict[cow]
    for frame in cow_frames:
        frame_path = os.path.join(src_path, frame +".txt")
        shutil.copy(frame_path, dest_path)