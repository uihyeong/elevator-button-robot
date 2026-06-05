from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='yolo/datasets/BOX_v2/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    project='yolo/runs',
    name='box_v2',
    exist_ok=True,
)
