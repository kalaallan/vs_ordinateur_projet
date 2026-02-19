from ultralytics import YOLO
import os
import random
import cv2

# === CONFIG ===
YOLO11_PATH = r"runs/detect/train3/weights/best.pt"
YOLO12_PATH = r"runs/detect/yolo12/weights/best.pt"
YOLO26_PATH = r"runs/detect/yolo26/weights/best.pt"

SOURCE_DIR = r"valid/images"
OUT_DIR = r"visual_compare"
CONF = 0.6
IOU = 0.5

os.makedirs(OUT_DIR, exist_ok=True)


def predict_with_legacy_nms_if_needed(model, source, **kwargs):
    uses_end2end_head = bool(getattr(getattr(model, "model", None), "end2end", False))
    if not uses_end2end_head:
        return model.predict(source, **kwargs)

    try:
        return model.predict(source, end2end=False, **kwargs)
    except Exception as exc:
        if "end2end" in str(exc).lower():
            return model.predict(source, **kwargs)
        raise


# Charger modèles
model11 = YOLO(YOLO11_PATH)
model12 = YOLO(YOLO12_PATH)
model26 = YOLO(YOLO26_PATH)

# Prendre 10 images au hasard
images = [f for f in os.listdir(SOURCE_DIR) if f.endswith((".jpg", ".png", ".jpeg"))]
images = random.sample(images, min(10, len(images)))

for name in images:
    path = os.path.join(SOURCE_DIR, name)

    # YOLO11
    r11 = predict_with_legacy_nms_if_needed(model11, path, conf=CONF, iou=IOU, verbose=False)[0]
    im11 = r11.plot()

    # YOLO12
    r12 = predict_with_legacy_nms_if_needed(model12, path, conf=CONF, iou=IOU, verbose=False)[0]
    im12 = r12.plot()

    # YOLO26
    r26 = predict_with_legacy_nms_if_needed(model26, path, conf=CONF, iou=IOU, verbose=False)[0]
    im26 = r26.plot()

    # Sauvegarde
    cv2.imwrite(os.path.join(OUT_DIR, f"y11_{name}"), im11)
    cv2.imwrite(os.path.join(OUT_DIR, f"y12_{name}"), im12)
    cv2.imwrite(os.path.join(OUT_DIR, f"y26_{name}"), im26)

print("✅ Comparaison visuelle terminée ->", OUT_DIR)
