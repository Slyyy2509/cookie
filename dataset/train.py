import os
from pathlib import Path
from ultralytics import YOLO

# CWD всегда = директория скрипта, независимо от того, откуда запустили
os.chdir(Path(__file__).resolve().parent)

model = YOLO('best.pt')

# явный project = папка для runs/, рядом со скриптом
model.train(
    data=str(Path(__file__).resolve().parent / 'data.yaml'),
    epochs=100,
    imgsz=640,
    batch=16,
    workers=0,
    name='clicker_model_up',
    project=str(Path(__file__).resolve().parent / 'runs' / 'detect'),
    patience=50,
)