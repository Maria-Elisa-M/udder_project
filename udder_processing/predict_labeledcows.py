import pandas as pd
import os
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
from tifffile import imwrite

# good frames
dirpath = os.getcwd()
frames_df = pd.read_csv(os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r"udder_labels\frame_class_list.csv"))
good_frames = frames_df[frames_df.frame_class ==1]
good_frames = good_frames.reset_index()
image_path = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r"udder_video\depth_images")

# model path
model_path = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r'udder_models')
modelpath_classify = os.path.join(model_path, r"frame_classify\runs\classify\train\weights\best.pt")
modelpath_segment = os.path.join(model_path, r"udder_segment\runs\segment\train\weights\best.pt")
modelpath_keypoints = os.path.join(model_path, r"teat_keypoints\runs\pose\train\weights\best.pt")

model_classify = YOLO(modelpath_classify)
model_segment = YOLO(modelpath_segment)
model_keypoints = YOLO(modelpath_keypoints)


def save_segment(filename, polygon):
    outpath = os.path.join(r"validate_watershed\pred_labels\segments", filename)
    segment = [str(pt) for p in  polygon for pt in p]
    segment = [str(0)] + segment
    with open(outpath, "w") as f:
        f.write(" ".join(segment))

def save_keypoints(filename, kpoints, bbox):
    outpath = os.path.join(r"validate_watershed\pred_labels\keypoints", filename)
    points = [str(pt) for p in  kpoints for pt in p]
    points = [str(0)] + [str(p) for p in bbox] + points
    with open(outpath, "w") as f:
        f.write(" ".join(points))

def save_bbox(filename,  bbox):
    outpath = os.path.join(r"validate_watershed\pred_labels\bbox", filename)
    bbox = [str(0)] + [str(p) for p in bbox]
    with open(outpath, "w") as f:
        f.write(" ".join(bbox))

def is_not_dup(arr):
    u, c = np.unique(arr, axis=0, return_counts=True)
    return not (c>1).any()

def mask_img(poly, img):
    h, w = img.shape
    mask2 = np.zeros([h,w]).astype("int16")
    mask = cv2.fillPoly(mask2, np.array([poly]).astype(np.int32), color=1)
    masked_im = (img*mask).astype("int16")
    return masked_im


for i in range(len(good_frames)):
    line = good_frames.loc[i]
    cow = line["cow"]
    filename = line["filename"] + ".tif"
    filepath = os.path.join(image_path, str(cow), filename)
    outname =  line["filename"] + ".txt"
    img = np.asarray(Image.open(filepath)).astype("int16")
    results = model_classify(filepath)
    prob_array = results[0].probs.data.tolist()
    if prob_array[1] > 0.9:
        results = model_segment(filepath)
        if (len(results) > 0) & (results[0].masks is not None):
            polyn = (results[0].masks[0].xyn[0]).tolist()
            poly = (results[0].masks[0].xy[0]).tolist()
            bbox = (results[0].boxes.xywhn[0]).tolist()
            save_bbox(outname, bbox)
            save_segment(outname, polyn)
            masked_im = mask_img(poly, img)
            imwrite("temp_img.tif", masked_im)
            results = model_keypoints("temp_img.tif")
            if (len(results[0].keypoints.xyn[0]) > 0):
                kpoints = results[0].keypoints.xyn[0].tolist()
                kpoints2 = np.array(kpoints).reshape((4,2))
                kpoints = np.hstack((kpoints2, [[2]]*4)).tolist()
                os.remove("temp_img.tif")
                if is_not_dup(kpoints2):
                    save_keypoints(outname, kpoints, bbox)