# Maria Elisa Montes
# Working version: predict_class_ws_newcows
# last update: 2025-02-10

import os
import pandas as pd
from ultralytics import YOLO
import numpy as np 
from PIL import Image
import json


dirpath = os.getcwd()
config_path = os.path.join(dirpath, "udder_config.json")

# Open and read the JSON file
with open(config_path, 'r') as file:
    data = json.load(file)

def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# output_path 
output_path = data["temp_path"]
mk_dir(label_path)

# model path
model_path = data["model_path"]
model_path_ws = os.path.join(model_path, r'ws_classify\train2\weights\best.pt')
model = YOLO(model_path_ws)

# video path
input_path = data["temp_path"]
label_path = os.path.join(input_path, "pred_labels")
label_dir = os.path.join(label_path, "watershed_segments")
corr_dir = os.path.join(label_path, "watershed_correspondence")

file_list = [file.replace(".npy", "") for file in os.listdir(label_dir)]

color_dict = {"lf":[1,1,0], "rf": [0, 1, 1], "lb":[1, 0,1], "rb":[1,0,0], "bg": [0, 0, 0]}

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
predictions_df.to_csv(os.path.join(label_path, "ws_class_predictions_II.csv"), index = False)