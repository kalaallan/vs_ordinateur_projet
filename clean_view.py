from ultralytics import YOLO
import os

model = YOLO(r"runs/detect/train3/weights/best.pt")

source = r"valid/images"
outdir = r"clean_preds"
os.makedirs(outdir, exist_ok=True)

# classes à garder: Ball(0) et Player(3)
KEEP = [0, 3]


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


for name in os.listdir(source):
    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    path = os.path.join(source, name)

    r = predict_with_legacy_nms_if_needed(
        model, path, conf=0.5, iou=0.5, classes=KEEP, verbose=False
    )[0]

    # plot plus clean : pas de labels / pas de conf
    im = r.plot(labels=False, conf=False, line_width=2)

    # sauvegarde
    import cv2
    cv2.imwrite(os.path.join(outdir, name), im)

print("✅ Images sauvegardées dans", outdir)
