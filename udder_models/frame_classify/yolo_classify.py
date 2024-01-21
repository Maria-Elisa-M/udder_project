from ultralytics import YOLO
import os

if __name__ ==  '__main__':
    
    # Load a model
    model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
    
    # Train the model
    dirpath = os.path.join(os.getcwd(), "datasets")
    model.train(data=dirpath, epochs=100, imgsz=848)