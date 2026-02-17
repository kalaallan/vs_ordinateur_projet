from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,  # tu peux ajuster (ou mettre 0 si ça re-plante)
    )

    model.val(data="data.yaml", device=0)

    print("OK ✅ best.pt est dans runs/detect/train/weights/best.pt")


if __name__ == "__main__":
    freeze_support()
    main()
