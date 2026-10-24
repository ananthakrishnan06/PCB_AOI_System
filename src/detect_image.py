from ultralytics import YOLO
import cv2
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train-7" / "weights" / "best.pt"
IMAGE_FOLDER = PROJECT_ROOT / "dataset" / "images" / "test"

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("Model Loaded Successfully!")

# ============================================================
# LOAD TEST IMAGE
# ============================================================

images = list(IMAGE_FOLDER.glob("*.jpg"))

if len(images) == 0:
    images = list(IMAGE_FOLDER.glob("*.png"))

if len(images) == 0:
    raise FileNotFoundError("No images found inside dataset/images/test")

image = images[0]

print(f"\nTesting Image : {image.name}")

# ============================================================
# DETECTION
# ============================================================

results = model.predict(
    source=str(image),
    conf=0.25,
    save=False,
    verbose=False
)

result = results[0]

annotated = result.plot()

# ============================================================
# SAVE IMAGE
# ============================================================

save_path = OUTPUT_DIR / "prediction.jpg"

cv2.imwrite(str(save_path), annotated)

print(f"\nAnnotated image saved to:\n{save_path}")

# ============================================================
# INSPECTION REPORT
# ============================================================

boxes = result.boxes

print("\n")
print("=" * 95)
print("PCB INSPECTION REPORT".center(95))
print("=" * 95)

print(f"\nImage : {image.name}")

if len(boxes) == 0:

    print("\nNo defects detected.")
    print("\nSTATUS : PASS")

else:

    defect_summary = {}

    print("\nDetected Defects\n")

    print(
        f"{'ID':<5}"
        f"{'Defect':<22}"
        f"{'Confidence':<14}"
        f"{'Center(X,Y)':<18}"
        f"{'Bounding Box'}"
    )

    print("-" * 95)

    for idx, box in enumerate(boxes, start=1):

        cls = int(box.cls[0])

        conf = float(box.conf[0])

        name = model.names[cls]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if name not in defect_summary:
            defect_summary[name] = 0

        defect_summary[name] += 1

        print(
            "{:<5}{:<22}{:<14}{:<18}({}, {}) ({}, {})".format(
                idx,
                name,
                f"{conf:.2%}",
                f"({cx},{cy})",
                x1,
                y1,
                x2,
                y2,
            )
        )

    print("-" * 95)

    print(f"\nTotal Defects : {len(boxes)}")

    print("\nDefect Summary")
    print("-" * 30)

    for defect, count in defect_summary.items():
        print(f"{defect:<22}: {count}")

    print("\nSTATUS : FAIL")

print("=" * 95)

# ============================================================
# SHOW IMAGE
# ============================================================

cv2.imshow("PCB AOI Detection", annotated)

cv2.waitKey(0)

cv2.destroyAllWindows()