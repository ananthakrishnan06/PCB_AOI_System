from detector import detect_image
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE = PROJECT_ROOT / "dataset" / "images" / "test"

images = list(IMAGE.glob("*.jpg"))

if len(images) == 0:
    images = list(IMAGE.glob("*.png"))

result = detect_image(str(images[0]))

print("\nStatus :", result["status"])

print("Total Defects :", result["total"])

print("\nSummary")

print(result["summary"])

cv2.imshow("Detection", result["annotated"])

cv2.waitKey(0)

cv2.destroyAllWindows()