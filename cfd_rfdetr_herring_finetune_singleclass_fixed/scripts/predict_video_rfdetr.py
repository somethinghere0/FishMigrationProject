import argparse
from pathlib import Path
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm


def load_model(model_size: str, checkpoint: str):
    try:
        from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
    except Exception as exc:
        raise ImportError(f"Could not import rfdetr. Install requirements first. Original error: {exc}")
    cls_map = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium}
    if model_size not in cls_map:
        raise ValueError("model_size must be nano, small, or medium")
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return cls_map[model_size](pretrain_weights=checkpoint)


def detections_to_xyxy_conf_class(detections):
    # RF-DETR returns a supervision Detections object in current examples.
    xyxy = getattr(detections, "xyxy", [])
    conf = getattr(detections, "confidence", None)
    class_id = getattr(detections, "class_id", None)
    if conf is None:
        conf = [1.0] * len(xyxy)
    if class_id is None:
        class_id = [0] * len(xyxy)
    return xyxy, conf, class_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="videos/Herring.mp4")
    ap.add_argument("--checkpoint", default="outputs/cfd_herring_finetune/checkpoint_best_total.pth")
    ap.add_argument("--model-size", default="nano", choices=["nano", "small", "medium"])
    ap.add_argument("--out-video", default="outputs/predicted_herring_rfdetr.mp4")
    ap.add_argument("--out-csv", default="outputs/predicted_herring_rfdetr.csv")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--frame-stride", type=int, default=1, help="Run every N frames; 1 = every frame")
    args = ap.parse_args()

    model = load_model(args.model_size, args.checkpoint)
    video = Path(args.video)
    if not video.exists():
        raise FileNotFoundError(video)
    Path(args.out_video).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(args.out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    rows = []
    last_boxes = []
    last_scores = []
    last_class_ids_raw = []
    for frame_idx in tqdm(range(total), desc="Predicting"):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, args.frame_stride) == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            det = model.predict(pil, threshold=args.threshold)
            boxes, scores, class_ids = detections_to_xyxy_conf_class(det)
            last_boxes = boxes
            last_scores = scores
            last_class_ids_raw = class_ids
        for box, score, raw_cid in zip(last_boxes, last_scores, last_class_ids_raw):
            x1, y1, x2, y2 = [int(v) for v in box]
            # This project is single-class fish counting.
            # RF-DETR may expose raw IDs from the checkpoint; for display/counting we force class 0 = fish.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"fish {float(score):.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            try:
                raw_cid_value = int(raw_cid)
            except Exception:
                raw_cid_value = str(raw_cid)
            rows.append({"frame": frame_idx, "class_id_raw": raw_cid_value, "class_id_used": 0, "class_name": "fish", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": float(score)})
        cv2.putText(frame, f"Visible fish detections: {len(last_boxes)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        writer.write(frame)

    cap.release()
    writer.release()
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Saved annotated video: {args.out_video}")
    print(f"Saved detection CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
