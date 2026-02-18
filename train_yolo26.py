from ultralytics import YOLO
from multiprocessing import freeze_support
import torch


def val_with_legacy_nms_if_needed(model, **kwargs):
    """Keep YOLO26 validation output compatible with YOLO11/12 flow."""
    uses_end2end_head = bool(getattr(getattr(model, "model", None), "end2end", False))
    if not uses_end2end_head:
        return model.val(**kwargs)

    try:
        return model.val(end2end=False, **kwargs)
    except Exception as exc:
        if "end2end" in str(exc).lower():
            return model.val(**kwargs)
        raise


def main():
    if torch.cuda.is_available():
        device = 0
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    workers = 0 if device in ("cpu", "mps") else 4
    print(f"Device auto-detected: {device}")

    model = YOLO("yolo26n.pt")

    model.train(
        data="data.yaml",
        epochs=10,
        imgsz=640,
        batch=5,
        device=device,
        workers=workers,
        name="yolo26",
    )

    val_with_legacy_nms_if_needed(model, data="data.yaml", device=device)

    print("OK ✅ best.pt est dans runs/detect/yolo26/weights/best.pt")


if __name__ == "__main__":
    freeze_support()
    main()
