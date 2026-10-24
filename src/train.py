from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        device=0,
        workers=0
    )

if __name__ == "__main__":
    main()