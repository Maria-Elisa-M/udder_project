# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 09:45:47 2023

@author: Maria
"""
from ultralytics import YOLO

if __name__ ==  '__main__':
  model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
  model.train(data= r"C:\Users\Maria\Documents\udder_project_3\keypoints\config.yaml", epochs=100, imgsz=864, degrees = 180, scale = 0.5)
  print('yolo done!')