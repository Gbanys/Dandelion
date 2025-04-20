from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import numpy as np

app = FastAPI()
model = YOLO("yolov8s.pt")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    img_data = await image.read()
    img = Image.open(BytesIO(img_data)).convert("RGB")
    results = model(img, imgsz=416)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = model.names[int(box.cls[0])]
        conf = float(box.conf[0])

        detections.append({
            "label": label,
            "confidence": conf,
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
        })

    return {"detections": detections}
