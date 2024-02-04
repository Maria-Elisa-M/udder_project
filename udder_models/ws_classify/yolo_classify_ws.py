# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:02:10 2024

@author: marie
"""

from ultralytics import YOLO
import os

if __name__ ==  '__main__':
    
    # Load a model
    model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
    
    # Train the model
    dirpath = os.path.join(os.getcwd(), "ws_mask\data")
    model.train(data=dirpath, epochs=100, imgsz=864, degrees = 180, scale = 0.5)