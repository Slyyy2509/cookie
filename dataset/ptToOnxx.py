from ultralytics import YOLO

model = YOLO("D:/Тимур-аи/andrey-repo/cookie/dataset/best.pt")
model.export(format="onnx", opset=12, simplify=True)