# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 12:48:48 2024

@author: marie
"""

from ultralytics import YOLO
import os
import pandas as pd
import re
import numpy as np

dir_path = os.getcwd()
sets_df = pd.read_csv("wsclass_sets.csv")


all_preds = pd.DataFrame()
all_cfm = pd.DataFrame()
for run in range(5):
    run_name = "data_k" + str(run)
    model_name = "model_k" + str(run)
    run_dir = os.path.join(dir_path,"ws_mask", "folds", run_name)
    model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
    
    # Train the model
    datapath = os.path.join(os.getcwd(), run_dir)
    model.train(data=datapath, epochs=100, imgsz=864, degrees = 180, scale = 0.5)
    
    test_images = sets_df[(sets_df.set_name == "test") &  (sets_df.kfold == run)].reset_index()
    os.rename("runs", model_name)
    model_path = os.path.join(model_name, 'classify/train/weights/best.pt')
    model = YOLO(model_path)
    
    #%% image dir
    data_dir = os.path.join(run_dir, "test")

    clas_dict = {0:"bad", 1:"good"}

    pred_df = pd.DataFrame({"filename": test_images.filename,\
                            "img_class": test_images.img_class,\
                            "p_bad": [-1]*len( test_images),\
                            "p_good":[-1]*len( test_images),\
                            "argmax":[-1]*len( test_images),\
                            "thr08": [-1]*len( test_images),\
                            "thr05": [-1]*len( test_images),\
                            "thr09": [-1]*len( test_images),\
                           "fold": [run]*len( test_images)})
    #%%                            
    # for image in tests
    for file in test_images.filename:
        file_name = file + ".png"
        file_line = test_images[test_images.filename == file]
        class_dir = clas_dict[file_line.img_class.values[0]]
        # find source directory
        img_dir = os.path.join(data_dir,class_dir, file_name)
        # get prediction on image
        results = model(img_dir)
        prob_array = results[0].probs.data.tolist()
        pred_df.loc[pred_df.filename == file, "p_bad"] = prob_array[0]
        pred_df.loc[pred_df.filename == file, "p_good"] = prob_array[1]
        pred_df.loc[pred_df.filename == file, "argmax"] =  np.argmax(prob_array)
        pred_df.loc[pred_df.filename == file, "thr05"] = [1 if prob_array[1]> 0.5 else 0][0]
        pred_df.loc[pred_df.filename == file, "thr08"] = [1 if prob_array[1]> 0.8 else 0][0]
        pred_df.loc[pred_df.filename == file, "thr09"] = [1 if prob_array[1]> 0.9 else 0][0]
    all_preds = pd.concat([all_preds, pred_df], axis=0, ignore_index=True)
    #%%
    cf_mat = pd.DataFrame(columns = ['pred_img_class', 'pred_0', 'pred_1', 'total', 'thr', "fold"])
    for thr in ["argmax", "thr05", "thr08", "thr09"]:
        df = pred_df[[thr, "img_class", "filename"]].groupby(["img_class", thr]).agg("count").reset_index()
        df = df.pivot( index = "img_class", columns= thr, values = "filename").reset_index()
        df.columns = ["_".join(["pred", str(c)]) for c in df.columns]
        pred_cols = [col for col in df.columns if re.match(r"pred_\d{1}", col)]
        df["total"] = df.loc[:,pred_cols].sum(axis=1)
        df["thr"] = thr
        df["fold"] = run
        cf_mat = pd.concat([cf_mat, df], axis = 0, ignore_index=True)
    all_cfm = pd.concat([all_cfm, cf_mat], axis=0, ignore_index=True)
    
    #os.rmdir("runs")
all_preds.to_csv("ws_mask_classify_cross_predictions.csv", index = False)
all_cfm.to_csv("ws_mask_classify_cross_cfmatrix.csv",  index = False)