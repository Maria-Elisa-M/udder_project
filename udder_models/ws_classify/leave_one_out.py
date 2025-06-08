
from ultralytics import YOLO
import os
import pandas as pd
import numpy as np
import shutil

data_path =  r'C:\Users\marie\rep_codes\udder_project\udder_processing\validate_watershed\watershed_data\masked_frame'
working_path = r"C:\Users\marie\rep_codes\udder_project\udder_models\ws_classify\leave_one_out"
df = pd.read_csv(r"C:\Users\marie\rep_codes\udder_project\udder_processing\validate_watershed\survey_results.csv")
cow_list = np.unique(df.cow)
dataset_path_src = os.path.join(working_path, "dataset")

# create training dataset
clas_dict = {0:"bad", 1:"good"}

def create_dataset(train_files, test_files, dataset_path, source_path):
    os.makedirs(dataset_path, exist_ok = True)
    os.makedirs(os.path.join(dataset_path, "train", "good"), exist_ok = True)
    os.makedirs(os.path.join(dataset_path, "train", "bad"), exist_ok = True)
    os.makedirs(os.path.join(dataset_path, "test", "good"), exist_ok = True)
    os.makedirs(os.path.join(dataset_path, "test", "bad"), exist_ok = True)
    # move train files
    for i in range(len(train_files)):
        filename = train_files.iloc[i].filename
        img_class = clas_dict[train_files.iloc[i].img_class]
        src = os.path.join(source_path, f"{filename}.png")
        dst = os.path.join(dataset_path, "train", img_class, f"{filename}.png")
        shutil.copy(src, dst)
    for i in range(len(test_files)):
        filename = f"{test_files.iloc[i].filename}"
        img_class = clas_dict[test_files.iloc[i].img_class]
        src = os.path.join(source_path, f"{filename}.png")
        dst = os.path.join(dataset_path, "test", img_class, f"{filename}.png")
        shutil.copy(src, dst)
        

# tarin a model excluding one cow at onece
for i, cow in enumerate(cow_list[:1]):
    test_files = df[df.cow == cow].reset_index(drop = True)
    train_files = df[df.cow != cow].reset_index(drop = True)
    # move files to dataset location
    create_dataset(train_files, test_files, dataset_path_src, data_path)

    # remove file from the training set
    model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
      
    #Train the model
    model.train(data=dataset_path_src, epochs=100, imgsz=864, degrees = 180, scale = 0.5)
    
    # reaname directories
    model_path_src = os.apth.join(working_path, "runs")
    model_path_dst = os.apth.join(working_path, f"model_{cow}")
    dataset_path_dst = os.apth.join(working_path,f"dataset_{cow}")
    os.rename(model_path_src, model_path_dst)
    os.rename(dataset_path_src, dataset_path_dst)
    # predict on the test images
    model_full_path = os.path.join(model_path_dst, 'classify/train/weights/best.pt')
    modelk = YOLO(model_full_path)
    
    # save predictions


df = pred_df.melt(id_vars = ["kfold","img_class"], value_vars = ["argmax","thr08","thr05","thr09"])
grouped = df.groupby(["kfold","variable", "img_class"]).agg(["count", "sum"]).reset_index()
grouped.columns = ["fold","variable", "img_class", "total", "pred_1"]
grouped["pred_0"] = grouped["total"]-grouped["pred_1"]


pred_df.to_csv("ws_mask_classify_cross_predictions.csv", index = False)
grouped.to_csv("ws_mask_classify_cross_cfmatrix.csv",  index = False)