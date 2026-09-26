import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trained YOLO model on herring video with optional tracking.")
    parser.add_argument("--weights", default="runs/train/herring_yolo_finetune/weights/best.pt")
    parser.add_argument("--source", default="videos/Herring.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tracker", default="botsort.yaml", help="botsort.yaml or bytetrack.yaml")
    parser.add_argument("--project", default="runs/predict")
    parser.add_argument("--name", default="herring_video_prediction")
    parser.add_argument("--no-track", action="store_true", help="Use normal prediction instead of tracking")
    args = parser.parse_args()

    if not Path(args.source).exists():
        raise FileNotFoundError(f"Video not found: {args.source}")
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    device = choose_device(args.device)
    print(f"Using device: {device}")
    model = YOLO(args.weights)

    if args.no_track:
        model.predict(
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            save=True,
            project=args.project,
            name=args.name,
        )
    else:
        model.track(
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            tracker=args.tracker,
            save=True,
            persist=True,
            project=args.project,
            name=args.name,
        )

    print("Prediction complete.")
    print(f"Saved outputs under: {Path(args.project) / args.name}")
