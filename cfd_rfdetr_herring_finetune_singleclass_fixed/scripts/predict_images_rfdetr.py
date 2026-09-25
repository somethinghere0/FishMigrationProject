import argparse
from pathlib import Path
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm


def load_model(model_size: str, checkpoint: str):
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
    cls_map = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium}
    return cls_map[model_size](pretrain_weights=checkpoint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="frames_raw/Herring")
    ap.add_argument("--checkpoint", default="outputs/cfd_herring_finetune/checkpoint_best_total.pth")
    ap.add_argument("--model-size", default="nano", choices=["nano", "small", "medium"])
    ap.add_argument("--out", default="outputs/predicted_images")
    ap.add_argument("--csv", default="outputs/predicted_images.csv")
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()
    images_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model_size, args.checkpoint)
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    rows = []
    for p in tqdm(image_paths, desc="Predict images"):
        img = Image.open(p).convert("RGB")
        det = model.predict(img, threshold=args.threshold)
        xyxy = getattr(det, "xyxy", [])
        conf = getattr(det, "confidence", None)
        raw_class_ids = getattr(det, "class_id", None)
        if conf is None:
            conf = [1.0] * len(xyxy)
        if raw_class_ids is None:
            raw_class_ids = [0] * len(xyxy)
        frame = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for i, (box, score, raw_cid) in enumerate(zip(xyxy, conf, raw_class_ids)):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            try:
                raw_cid_value = int(raw_cid)
            except Exception:
                raw_cid_value = str(raw_cid)
            rows.append({"image": p.name, "det_id": i, "class_id_raw": raw_cid_value, "class_id_used": 0, "class_name": "fish", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": float(score)})
        cv2.putText(frame, f"fish: {len(xyxy)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        cv2.imwrite(str(out_dir / p.name), frame)
    pd.DataFrame(rows).to_csv(args.csv, index=False)
    print(f"Saved annotated images to {out_dir}")
    print(f"Saved CSV to {args.csv}")

if __name__ == "__main__":
    main()
