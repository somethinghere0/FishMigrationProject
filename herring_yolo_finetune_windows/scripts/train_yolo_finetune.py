import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"  # NVIDIA GPU, common on Windows/lab machines
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon Mac GPU
    return "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the last layers of a pretrained YOLO model for herring detection.")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained model, e.g., yolov8n.pt, yolo11n.pt, or old current_fish_model.pt")
    parser.add_argument("--data", default="configs/herring.yaml", help="YOLO dataset YAML")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--freeze", type=int, default=10, help="Freeze first N layers so mostly later layers/head train")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 1, mps")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="herring_yolo_finetune")
    parser.add_argument("--workers", type=int, default=0, help="Use 0 on Windows to avoid multiprocessing issues")
    args = parser.parse_args()

    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading pretrained model: {args.model}")
    print(f"Freezing first {args.freeze} layers. This fine-tunes mainly later layers/head.")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        device=device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=15,
        optimizer="auto",
        cos_lr=True,
        cache=False,
        verbose=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    last = Path(args.project) / args.name / "weights" / "last.pt"
    print("Training complete.")
    print(f"Best weights expected at: {best}")
    print(f"Last weights expected at: {last}")
