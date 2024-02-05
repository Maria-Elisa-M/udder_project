# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 14:29:47 2024

@author: marie
"""
import os
import pandas as pd
from ultralytics import YOLO
import numpy as np 
from PIL import Image
dirpath = os.getcwd()
# model
model_path = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r'udder_models\ws_classify\runs\classify\train2\weights\best.pt')
model = YOLO(model_path)

color_dict = {"lf":[1,1,0], "rf": [0, 1, 1], "lb":[1, 0,1], "rb":[1,0,0], "bg": [0, 0, 0]}

label_dir = r"validate_watershed\watershed_segments"
corr_dir = r"validate_watershed\watershed_correspondence"
# file_list = os.listdir(label_dir)
df = pd.read_csv(r"validate_watershed\survey_groups.csv")
file_list = [file.replace(".npy", "") for file in os.listdir(label_dir)]
#%%
predictions_df = pd.DataFrame()
cnt = 0
for file in file_list:
    label_file = os.path.join(label_dir, file+".npy")
    labels = np.load(label_file)
    kp_ws = pd.read_csv(os.path.join(corr_dir, file +".csv")).loc[0].to_dict()
    ws_map = dict((v, k) for k, v in kp_ws.items())
    ws_map[0] = "bg"
    labels_img = np.zeros((labels.shape[0], labels.shape[1], 3))
    for key in ws_map.keys():
        quarter_mask = labels.copy()
        quarter_mask[labels !=key] = 0
        rows,cols = np.nonzero(quarter_mask)
        labels_img[rows, cols, :] = color_dict[ws_map[key]]
    im = Image.fromarray(np.uint8(labels_img)*255, 'RGB')
    results = model(im)
    prob_array = results[0].probs.data.tolist()
    # store predictions
    line = pd.DataFrame({"filename": file,\
                        "p_bad": prob_array[0],\
                        "p_good":prob_array[1],\
                        "argmax": np.argmax(prob_array),\
                        "thr08": [1 if prob_array[1]> 0.5 else 0][0],\
                        "thr05":[1 if prob_array[1]> 0.8 else 0][0],\
                        "thr09": [1 if prob_array[1]> 0.9 else 0][0]},
                        index = [0])
    print(f"{cnt}: {file}")
    cnt +=1
    
    predictions_df = pd.concat([predictions_df, line], axis = 0, ignore_index=True)
#%%
cows = [file.split("_")[0] for file in predictions_df.filename]
predictions_df["cow"] = cows
predictions_df.to_csv("validate_watershed\ws_class_predictions_I.csv", index = False)
