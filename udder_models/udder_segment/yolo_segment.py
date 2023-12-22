from ultralytics import YOLO

if __name__ ==  '__main__':
    model = YOLO('YOLOv8n-seg.pt')
    model.train(data= r"C:\Users\Maria\Documents\udder_project_3\segement\config.yaml", epochs=100, imgsz=864, degrees = 180, scale = 0.5)
    print('yolo done!')