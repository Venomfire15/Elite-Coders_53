from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  

model.train(data="configuration.yaml", epochs=1, save=True)  
