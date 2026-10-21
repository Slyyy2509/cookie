from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    workers=0,
    name='clicker_model',
    patience=50
)