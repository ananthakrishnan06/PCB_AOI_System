from pathlib import Path

from detector import detect_image
from report import generate_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_FOLDER = PROJECT_ROOT / "dataset" / "images" / "test"

images = list(IMAGE_FOLDER.glob("*.jpg"))

if not images:
    images = list(IMAGE_FOLDER.glob("*.png"))

image_path = images[0]

result = detect_image(str(image_path))

text = generate_report(result, image_path.name)

print(text)