from pathlib import Path
import yaml

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset folder
DATASET_PATH = PROJECT_ROOT / "dataset"

yaml_file = DATASET_PATH / "data.yaml"

if not yaml_file.exists():
    print("❌ data.yaml not found!")
    exit()

with open(yaml_file, "r") as f:
    data = yaml.safe_load(f)

print("=" * 50)
print("PCB DATASET EXPLORER")
print("=" * 50)

print(f"Dataset: {DATASET_PATH}")

print("\nClasses:")
for class_id, class_name in data["names"].items():
    print(f"{class_id}: {class_name}")

print(f"\nTotal Classes: {len(data['names'])}")

print("\nTrain:", data["train"])
print("Validation:", data["val"])
print("Test:", data["test"])