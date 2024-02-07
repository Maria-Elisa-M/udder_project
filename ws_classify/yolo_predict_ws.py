# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd
from ultralytics import YOLO
import numpy as np 
import re

# model
model_path = r'runs\classify\train3\weights\best.pt'
model = YOLO(model_path)

# test images
dirpath = os.getcwd()
frame_df = pd.read_csv(os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), "wsclass_sets.csv"))
test_images = frame_df[frame_df.set_name == "test"].reset_index()

#%% image dir
new_dir = os.path.join(os.path.sep.join(dirpath.split(os.path.sep)[:-2]), r'udder_video\depth_images')
old_dir = os.path.join(os.path.sep.join(dirpath.split(os.path.sep)[:-2]),  r'udder_dcc\images')
data_dir = os.path.join(dirpath, r"masked_frame\test")

clas_dict = {0:"bad", 1:"good"}

pred_df = pd.DataFrame({"filename": test_images.filename,\
                        "img_class": test_images.img_class,\
                        "p_bad": [-1]*len( test_images),\
                        "p_good":[-1]*len( test_images),\
                        "argmax":[-1]*len( test_images),\
                        "thr08": [-1]*len( test_images),\
                        "thr05": [-1]*len( test_images),\
                        "thr09": [-1]*len( test_images)})
#%%                            
# for image in tests
for file in test_images.filename:
    file_name = file + ".png"
    file_line = test_images[test_images.filename == file]
    # find source directory
    img_dir = os.path.join(data_dir, file_name)
    # get prediction on image
    results = model(img_dir)
    prob_array = results[0].probs.data.tolist()
    pred_df.loc[pred_df.filename == file, "p_bad"] = prob_array[0]
    pred_df.loc[pred_df.filename == file, "p_good"] = prob_array[1]
    pred_df.loc[pred_df.filename == file, "argmax"] =  np.argmax(prob_array)
    pred_df.loc[pred_df.filename == file, "thr05"] = [1 if prob_array[1]> 0.5 else 0][0]
    pred_df.loc[pred_df.filename == file, "thr08"] = [1 if prob_array[1]> 0.8 else 0][0]
    pred_df.loc[pred_df.filename == file, "thr09"] = [1 if prob_array[1]> 0.9 else 0][0]

#%%
cf_mat = pd.DataFrame(columns = ['pred_img_class', 'pred_0', 'pred_1', 'total', 'thr'])
for thr in ["argmax", "thr05", "thr08", "thr09"]:
    df = pred_df[[thr, "img_class", "filename"]].groupby(["img_class", thr]).agg("count").reset_index()
    df = df.pivot( index = "img_class", columns= thr, values = "filename").reset_index()
    df.columns = ["_".join(["pred", str(c)]) for c in df.columns]
    pred_cols = [col for col in df.columns if re.match(r"pred_\d{1}", col)]
    df["total"] = df.loc[:,pred_cols].sum(axis=1)
    df["thr"] = thr
    cf_mat = pd.concat([cf_mat, df], axis = 0, ignore_index=True)


pred_df.to_csv("ws_masked_classify_test_predictions.csv", index = False)
cf_mat.to_csv("ws_masked_classify_test_cfmatrix.csv")