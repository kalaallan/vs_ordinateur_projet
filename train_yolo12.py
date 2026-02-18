from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    model = YOLO("yolo12n.pt")  # modèle YOLO12 nano

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        name="yolo12",
    )

    model.val(data="data.yaml", device=0)


if __name__ == "__main__":
    freeze_support()
    main()
