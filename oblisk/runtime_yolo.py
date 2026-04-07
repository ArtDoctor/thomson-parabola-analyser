import logging
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from pydantic import BaseModel
from ultralytics import YOLO

from oblisk.config import default_yolo_detector_path

os.environ["YOLO_VERBOSE"] = "False"
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/oblisk-yolo")
ort.set_default_logger_severity(3)
logging.getLogger("ultralytics").setLevel(logging.ERROR)


class Coords(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


def cut_detector_image(
    image: np.ndarray,
    model_path: Path | None = None,
) -> Coords:
    model = YOLO(str(model_path or default_yolo_detector_path()), task="detect")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    results = model.predict(source=image, conf=0.5, verbose=False)
    boxes = results[0].boxes
    if boxes is None:
        return Coords(
            x1=0,
            y1=0,
            x2=image.shape[1],
            y2=image.shape[0],
            score=0.0,
        )
    coords = boxes.xyxy
    if len(coords) == 0:
        return Coords(
            x1=0,
            y1=0,
            x2=image.shape[1],
            y2=image.shape[0],
            score=0.0,
        )
    score_val = float(boxes.conf[0].item())
    x1, y1 = int(coords[0][0]), int(coords[0][1])
    x2, y2 = int(coords[0][2]), int(coords[0][3])

    img_h, img_w = image.shape[:2]
    box_area = (x2 - x1) * (y2 - y1)
    img_area = img_h * img_w
    min_area_ratio = 0.10
    if img_area > 0 and box_area / img_area < min_area_ratio:
        return Coords(x1=0, y1=0, x2=img_w, y2=img_h, score=0.0)

    return Coords(x1=x1, y1=y1, x2=x2, y2=y2, score=score_val)
