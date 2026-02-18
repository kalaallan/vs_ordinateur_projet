import cv2
import os
import random
import numpy as np
from ultralytics import YOLO

IMG_DIR = "valid/images"
TMP_DIR = "robust_tmp"
os.makedirs(TMP_DIR, exist_ok=True)

MODELS = {
    "YOLO11": r"runs/detect/train3/weights/best.pt",
    "YOLO12": r"runs/detect/yolo12/weights/best.pt",
}

PLAYER = 3
BALL = 0


def distort(img, mode):
    if mode == "blur":
        return cv2.GaussianBlur(img, (11, 11), 0)
    if mode == "dark":
        return (img * 0.5).astype(np.uint8)
    if mode == "noise":
        n = np.random.normal(0, 25, img.shape).astype(np.uint8)
        return cv2.add(img, n)
    if mode == "zoom":
        h, w, _ = img.shape
        crop = img[h // 6: -h // 6, w // 6: -w // 6]
        return cv2.resize(crop, (w, h))
    return img


images = random.sample(os.listdir(IMG_DIR), 20)
modes = ["normal", "blur", "dark", "noise", "zoom"]

for name, path in MODELS.items():
    print("\n", name)
    model = YOLO(path)

    for mode in modes:
        ball = 0
        player = 0

        for img_name in images:
            img = cv2.imread(os.path.join(IMG_DIR, img_name))
            if mode != "normal":
                img = distort(img, mode)

            tmp = os.path.join(TMP_DIR, "tmp.jpg")
            cv2.imwrite(tmp, img)

            r = model.predict(tmp, conf=0.5, verbose=False)[0]

            if r.boxes is not None:
                for b in r.boxes.cls:
                    if int(b) == BALL:
                        ball += 1
                    if int(b) == PLAYER:
                        player += 1

        print(f"{mode}: Ball={ball} Player={player}")
