import cv2
import yaml
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = PROJECT_ROOT / "dataset"

IMAGE_PATH = DATASET / "images" / "train"
LABEL_PATH = DATASET / "labels" / "train"

OUTPUT = PROJECT_ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

# -----------------------------
# Read class names
# -----------------------------
with open(DATASET / "data.yaml", "r") as f:
    data = yaml.safe_load(f)

class_names = data["names"]

# -----------------------------
# Load first image
# -----------------------------
image_files = list(IMAGE_PATH.glob("*.jpg"))

if not image_files:
    image_files = list(IMAGE_PATH.glob("*.png"))

if not image_files:
    print("No images found.")
    exit()

image_file = image_files[0]

print("Image:", image_file.name)

image = cv2.imread(str(image_file))

h, w = image.shape[:2]

label_file = LABEL_PATH / (image_file.stem + ".txt")

# -----------------------------
# Draw labels
# -----------------------------
if label_file.exists():

    with open(label_file) as f:

        for line in f.readlines():

            cls, x, y, bw, bh = map(float, line.split())

            cls = int(cls)

            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)

            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)

            cv2.rectangle(image,
                          (x1, y1),
                          (x2, y2),
                          (0, 255, 0),
                          2)

            cv2.putText(image,
                        class_names[cls],
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)

# -----------------------------
# Save result
# -----------------------------
save_path = OUTPUT / "visualized.jpg"

cv2.imwrite(str(save_path), image)

print("Saved to:", save_path)

cv2.imshow("PCB Labels", image)

cv2.waitKey(0)

cv2.destroyAllWindows()