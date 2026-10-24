from ultralytics import YOLO
import cv2
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train-7" / "weights" / "best.pt"
IMAGE_FOLDER = PROJECT_ROOT / "dataset" / "images" / "test"

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO Model...")

model = YOLO(str(MODEL_PATH))

print("Model Loaded Successfully!\n")

# ============================================================
# LOAD FIRST TEST IMAGE
# ============================================================

images = list(IMAGE_FOLDER.glob("*.jpg"))

if len(images) == 0:
    images = list(IMAGE_FOLDER.glob("*.png"))

if len(images) == 0:
    raise FileNotFoundError("No test images found!")

image_path = images[0]

print("Testing Image :", image_path.name)

image = cv2.imread(str(image_path))

# ============================================================
# RUN DETECTION
# ============================================================

results = model.predict(
    source=str(image_path),
    conf=0.25,
    verbose=False
)

result = results[0]

# ============================================================
# DRAW DEFECT MAP
# ============================================================

legend = []

for i, box in enumerate(result.boxes, start=1):

    cls = int(box.cls[0])
    conf = float(box.conf[0])

    name = model.names[cls]

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Center Point
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Draw center marker
    cv2.circle(image, (cx, cy), 6, (0, 0, 255), -1)
    cv2.circle(image, (cx, cy), 12, (255, 255, 255), 2)

    # Draw line from point to label
    cv2.line(image, (cx, cy), (cx + 30, cy - 30), (255, 255, 255), 1)

    # Label
    label = f"{name} ({conf:.0%})"

    cv2.putText(
        image,
        label,
        (cx + 35, cy - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    legend.append(f"{i}. {name} - {conf:.1%}")

# ============================================================
# DRAW TITLE
# ============================================================

cv2.rectangle(image, (10, 10), (430, 45), (40, 40, 40), -1)

cv2.putText(
    image,
    "PCB DEFECT MAP",
    (20, 37),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 255, 0),
    2
)

# ============================================================
# DRAW LEGEND
# ============================================================

legend_height = 40 + len(legend) * 28

cv2.rectangle(
    image,
    (10, 60),
    (430, 60 + legend_height),
    (40, 40, 40),
    -1
)

cv2.putText(
    image,
    "Detected Defects",
    (20, 85),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 255),
    2
)

y = 115

for item in legend:

    cv2.putText(
        image,
        item,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    y += 28

# ============================================================
# STATUS
# ============================================================

status = "PASS"

status_color = (0, 255, 0)

if len(result.boxes) > 0:
    status = "FAIL"
    status_color = (0, 0, 255)

cv2.rectangle(
    image,
    (10, y + 10),
    (430, y + 55),
    (40, 40, 40),
    -1
)

cv2.putText(
    image,
    f"STATUS : {status}",
    (20, y + 42),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    status_color,
    2
)

# ============================================================
# SAVE
# ============================================================

save_path = OUTPUT_DIR / "pcb_defect_map.jpg"

cv2.imwrite(str(save_path), image)

print("\nSaved to:")
print(save_path)

# ============================================================
# SHOW IMAGE
# ============================================================

cv2.imshow("PCB AOI - Defect Map", image)

cv2.waitKey(0)

cv2.destroyAllWindows()