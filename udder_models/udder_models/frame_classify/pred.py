# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd
from ultralytics import YOLO
import numpy as np 


# model
model_path = r'runs\classify\train\weights\best.pt'
model = YOLO(model_path)

# test images
dirpath = os.getcwd()
labels = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r'frameclass_sets.csv')
frame_df = pd.read_csv(labels)
test_images = frame_df[frame_df.set_name == "test"].reset_index()

# image dir
new_dir = os.path.join(os.path.sep.join(dirpath.split(os.path.sep)[:-2]), r'udder_video\depth_images')
old_dir = os.path.join(os.path.sep.join(dirpath.split(os.path.sep)[:-2]),  r'udder_dcc\images')
data_dir = os.path.join(dirpath, r"frame_classify\data")

# data collection groups
imgdir_dict = {20210625:{"lab": old_dir}, \
              20211022: {"lab": old_dir}, \
              20231117:{"guilherme": new_dir , \
                        "maria": new_dir}}

clas_dict = {0:"bad", 1:"good"}

pred_df = pd.DataFrame({"filename": test_images.filename,\
                        "frame_class": test_images.frame_class,\
                        "p_bad": [-1]*len( test_images),\
                        "p_good":[-1]*len( test_images),\
                        "argmax":[-1]*len( test_images),\
                        "thr08": [-1]*len( test_images),\
                        "thr05": [-1]*len( test_images),\
                        "thr09": [-1]*len( test_images)})
                            
# for image in tests
for file in test_images.filename:
    file_name = file + ".tif"
    file_line = test_images[test_images.filename == file]
    # find source directory
    file_date = file_line["date"].values[0]
    computer  = file_line["computer"].values[0]
    src_dir = os.path.join(imgdir_dict[file_date][computer], file_name.split("_")[0])
    img_dir = os.path.join(src_dir, file_name)
    # get prediction on image
    results = model(img_dir)
    prob_array = results[0].probs.data.tolist()
    pred_df.loc[pred_df.filename == file, "p_bad"] = prob_array[0]
    pred_df.loc[pred_df.filename == file, "p_good"] = prob_array[1]
    pred_df.loc[pred_df.filename == file, "argmax"] =  np.argmax(prob_array)
    pred_df.loc[pred_df.filename == file, "thr05"] = [1 if prob_array[1]> 0.5 else 0][0]
    pred_df.loc[pred_df.filename == file, "thr08"] = [1 if prob_array[1]> 0.8 else 0][0]
    pred_df.loc[pred_df.filename == file, "thr09"] = [1 if prob_array[1]> 0.9 else 0][0]


cf_mat = pd.DataFrame()
for thr in ["argmax", "thr05", "thr08", "thr09"]:
    df = pred_df[[thr,"filename", "frame_class"]].groupby(["frame_class", thr]).agg("count").reset_index()
    df = df.pivot( index = "frame_class", columns= thr, values = "filename").reset_index()
    df.columns = ["_".join(["pred", str(c)]) for c in df.columns]
    df["total"] = df.pred_1 +df.pred_0
    df["thr"] = thr
    cf_mat = pd.concat([cf_mat, df], axis = 0, ignore_index=True)


pred_df.to_csv("classify_test_predictions.csv", index = False)
cf_mat.to_csv("classify_test_cfmatrix.csv")