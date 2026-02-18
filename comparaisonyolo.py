from ultralytics import YOLO
import os
import random
import cv2

# === CONFIG ===
YOLO11_PATH = r"runs/detect/train3/weights/best.pt"
YOLO12_PATH = r"runs/detect/yolo12/weights/best.pt"

SOURCE_DIR = r"valid/images"
OUT_DIR = r"visual_compare"
CONF = 0.6
IOU = 0.5

os.makedirs(OUT_DIR, exist_ok=True)

# Charger modèles
model11 = YOLO(YOLO11_PATH)
model12 = YOLO(YOLO12_PATH)

# Prendre 10 images au hasard
images = [f for f in os.listdir(SOURCE_DIR) if f.endswith((".jpg", ".png", ".jpeg"))]
images = random.sample(images, min(10, len(images)))

for name in images:
    path = os.path.join(SOURCE_DIR, name)

    # YOLO11
    r11 = model11.predict(path, conf=CONF, iou=IOU, verbose=False)[0]
    im11 = r11.plot()

    # YOLO12
    r12 = model12.predict(path, conf=CONF, iou=IOU, verbose=False)[0]
    im12 = r12.plot()

    # Sauvegarde
    cv2.imwrite(os.path.join(OUT_DIR, f"y11_{name}"), im11)
    cv2.imwrite(os.path.join(OUT_DIR, f"y12_{name}"), im12)

print("✅ Comparaison visuelle terminée ->", OUT_DIR)
