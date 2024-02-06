# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 12:48:48 2024

@author: marie
"""

from ultralytics import YOLO
import os
import pandas as pd
import numpy as np

dir_path = os.getcwd()
sets_df = pd.read_csv(os.path.join(os.path.normpath(dir_path + os.sep + os.pardir), "wsclass_sets.csv"))

pred_df = sets_df.loc[sets_df.set_name == "test",["filename", "img_class", "kfold"]].drop_duplicates()
pred_df[["p_bad","p_good","argmax","thr08","thr05","thr09"]] = -1

clas_dict = {0:"bad", 1:"good"}
#%%
for run in range(5):
    run_name = "data_k" + str(run)
    model_name = "model_k" + str(run)
    run_dir = os.path.join(dir_path,"ws_mask", "folds", run_name)
    model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
      
    #Train the model
    datapath = os.path.join(os.getcwd(), run_dir)
    model.train(data=datapath, epochs=100, imgsz=864, degrees = 180, scale = 0.5)
    
    test_images = sets_df[(sets_df.set_name == "test") &  (sets_df.kfold == run)].reset_index()

    os.rename("runs", model_name)
    model_path = os.path.join(model_name, 'classify/train/weights/best.pt')
    modelk = YOLO(model_path)
    
    data_dir = os.path.join(run_dir, "test")
                   
    # for image in tests
    for file in test_images.filename:
        file_name = file + ".png"
        file_line = test_images[test_images.filename == file]
        class_dir = clas_dict[file_line.img_class.values[0]]
        # find source directory
        img_dir = os.path.join(data_dir,class_dir, file_name)
        # get prediction on image
        results = modelk(img_dir)
        prob_array = results[0].probs.data.tolist()
        pred_df.loc[pred_df.filename == file, "p_bad"] = prob_array[0]
        pred_df.loc[pred_df.filename == file, "p_good"] = prob_array[1]
        pred_df.loc[pred_df.filename == file, "argmax"] =  np.argmax(prob_array)
        pred_df.loc[pred_df.filename == file, "thr05"] = [1 if prob_array[1]> 0.5 else 0][0]
        pred_df.loc[pred_df.filename == file, "thr08"] = [1 if prob_array[1]> 0.8 else 0][0]
        pred_df.loc[pred_df.filename == file, "thr09"] = [1 if prob_array[1]> 0.9 else 0][0]


df = pred_df.melt(id_vars = ["kfold","img_class"], value_vars = ["argmax","thr08","thr05","thr09"])
grouped = df.groupby(["kfold","variable", "img_class"]).agg(["count", "sum"]).reset_index()
grouped.columns = ["fold","variable", "img_class", "total", "pred_1"]
grouped["pred_0"] = grouped["total"]-grouped["pred_1"]


pred_df.to_csv("ws_mask_classify_cross_predictions.csv", index = False)
grouped.to_csv("ws_mask_classify_cross_cfmatrix.csv",  index = False)