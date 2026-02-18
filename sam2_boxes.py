from ultralytics import YOLO, SAM
import os, cv2

YOLO_MODEL = r"runs/detect/train3/weights/best.pt"  # ton best YOLO11
SAM_MODEL = "sam2.1_b.pt"  # SAM2
SOURCE_DIR = r"valid/images"
OUT_DIR = r"yolo_sam2_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ta classe Player = 3 (selon ton data.yaml)
PLAYER_CLS = 3

yolo = YOLO(YOLO_MODEL)
sam = SAM(SAM_MODEL)

for name in sorted(os.listdir(SOURCE_DIR)):
    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(SOURCE_DIR, name)

    # 1) détecter joueurs avec YOLO
    det = yolo.predict(path, conf=0.5, iou=0.5, classes=[PLAYER_CLS], verbose=False)[0]

    if det.boxes is None or len(det.boxes) == 0:
        # rien à segmenter
        continue

    # 2) convertir les bbox en liste [x1,y1,x2,y2] (SAM2 accepte bbox prompt)
    bboxes = det.boxes.xyxy.cpu().numpy().tolist()

    # 3) SAM2 segmentation guidée par ces bbox
    seg = sam(path, bboxes=bboxes)[0]

    # 4) rendu visuel (masques + bbox)
    im = seg.plot()
    cv2.imwrite(os.path.join(OUT_DIR, name), im)

print("✅ YOLO -> SAM2 terminé ->", OUT_DIR)
