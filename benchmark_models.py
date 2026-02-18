from ultralytics import YOLO
import time
import glob
import torch

IMG_DIR = r"valid/images"
IMGS = glob.glob(IMG_DIR + "/*.jpg")[:100]  # 100 images max

MODELS = {
    "YOLO11": r"runs/detect/train3/weights/best.pt",
    "YOLO12": r"runs/detect/yolo12/weights/best.pt",
}


def bench(model_path):
    model = YOLO(model_path)
    # warmup
    for _ in range(5):
        _ = model.predict(IMGS[0], imgsz=640, conf=0.5, verbose=False)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    for p in IMGS:
        _ = model.predict(p, imgsz=640, conf=0.5, verbose=False)

    t1 = time.time()
    total = t1 - t0
    fps = len(IMGS) / total

    mem = None
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    return fps, total / len(IMGS), mem


for name, path in MODELS.items():
    fps, sec_img, mem = bench(path)
    print(
        f"{name}: {fps:.2f} FPS | {sec_img*1000:.1f} ms/img | GPU max mem: {mem:.0f} MB"
    )
