from ultralytics import YOLO
from pathlib import Path
import cv2
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train-7" / "weights" / "best.pt"

model = YOLO(str(MODEL_PATH))
INFERENCE_DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_HALF_PRECISION = torch.cuda.is_available()

if USE_HALF_PRECISION:
    model.to("cuda:0")
    model.half()


def detect_image(image_path, confidence=0.25, return_annotated=True):

    results = model.predict(
        source=image_path,
        conf=confidence,
        verbose=False,
        device=INFERENCE_DEVICE,
    )

    result = results[0]

    annotated = result.plot() if return_annotated else None

    defects = []
    summary = {}

    h, w = result.orig_shape

    for box in result.boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        defects.append({
            "class": name,
            "confidence": round(conf * 100, 2),
            "center": (cx, cy),
            "bbox": (x1, y1, x2, y2)
        })

        summary[name] = summary.get(name, 0) + 1

    status = "PASS" if len(defects) == 0 else "FAIL"

    return {
        "annotated": annotated,
        "defects": defects,
        "summary": summary,
        "total": len(defects),
        "status": status,
        "image_width": w,
        "image_height": h,
    }
