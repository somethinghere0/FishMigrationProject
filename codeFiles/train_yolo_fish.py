"""Train or continue-training the existing SeniorProject YOLO fish detector.

This script is designed to be dropped into the existing SeniorProject fish_counting
repository. It adds the missing training/retraining pipeline and supports YOLO12
model scales: yolo12n.pt, yolo12s.pt, yolo12m.pt, and yolo12x.pt.

Typical use cases:
    1) Train/fine-tune a YOLO12 baseline on the existing train/valid folders.
    2) Continue from the student's collected .pt weights.
    3) Train several YOLO12 scales for comparison.

Examples:
    python codeFiles/train_yolo_fish.py --model yolo12n.pt --epochs 50 --imgsz 640 --batch 8
    python codeFiles/train_yolo_fish.py --model yolo12s.pt --epochs 50 --batch 4
    python codeFiles/train_yolo_fish.py --model current_fish_model.pt --epochs 50 --freeze 10
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import yaml

YOLO12_MODELS = ["yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12x.pt"]


def project_root() -> Path:
    # script lives in codeFiles/; root is one level up
    return Path(__file__).resolve().parents[1]


def validate_dataset(data_yaml: Path) -> None:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing dataset YAML: {data_yaml}")
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Could not parse dataset YAML: {data_yaml}")
    root = (data_yaml.parent / cfg.get("path", ".")).resolve()
    train_img = (root / cfg["train"]).resolve()
    val_img = (root / cfg["val"]).resolve()
    if not train_img.exists():
        raise FileNotFoundError(f"Missing train images folder: {train_img}")
    if not val_img.exists():
        raise FileNotFoundError(f"Missing valid images folder: {val_img}")
    train_labels = Path(str(train_img).replace(os.sep + "images", os.sep + "labels"))
    val_labels = Path(str(val_img).replace(os.sep + "images", os.sep + "labels"))
    if not train_labels.exists():
        raise FileNotFoundError(f"Missing train labels folder: {train_labels}")
    if not val_labels.exists():
        raise FileNotFoundError(f"Missing valid labels folder: {val_labels}")


def resolve_model_arg(model_name: str, root: Path) -> str:
    """Allow local .pt files or Ultralytics model names like yolo12n.pt."""
    p = Path(model_name)
    if p.exists():
        return str(p.resolve())
    p_root = root / model_name
    if p_root.exists():
        return str(p_root.resolve())
    # If not local, let Ultralytics download/resolve it.
    return model_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain/fine-tune existing YOLO fish model with YOLO12 support")
    parser.add_argument(
        "--model",
        default="yolo12n.pt",
        help=(
            "Starting model .pt file. Recommended YOLO12 choices: yolo12n.pt, "
            "yolo12s.pt, yolo12m.pt, yolo12x.pt. You may also pass the old "
            "SeniorProject weights such as current_fish_model.pt or extraLarge.pt."
        ),
    )
    parser.add_argument("--data", default="fish_data.yaml", help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="0 for first NVIDIA GPU, cpu for CPU, mps for Apple Silicon")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default=None, help="Run name. If omitted, name is based on the model file.")
    parser.add_argument("--freeze", type=int, default=10, help="Freeze first N layers for last-layer fine-tuning. Use 0 to train all layers.")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted Ultralytics run")
    parser.add_argument("--cache", action="store_true", help="Cache dataset images in RAM/disk if memory allows")
    parser.add_argument("--amp", action="store_true", default=True, help="Use mixed precision when supported")
    args = parser.parse_args()

    root = project_root()
    os.chdir(root)
    sys.path.insert(0, str(root))

    data_yaml = (root / args.data).resolve()
    validate_dataset(data_yaml)

    from ultralytics import YOLO

    model_arg = resolve_model_arg(args.model, root)
    run_name = args.name
    if run_name is None:
        run_name = f"herring_{Path(args.model).stem}_retrain"

    print("=" * 80)
    print("SeniorProject YOLO fish retraining")
    print(f"Project root : {root}")
    print(f"Model start  : {model_arg}")
    print(f"YOLO12 set   : {', '.join(YOLO12_MODELS)}")
    print(f"Data YAML    : {data_yaml}")
    print(f"Epochs       : {args.epochs}")
    print(f"Image size   : {args.imgsz}")
    print(f"Batch        : {args.batch}")
    print(f"Device       : {args.device}")
    print(f"Freeze       : {args.freeze}")
    print(f"Run name     : {run_name}")
    print("=" * 80)

    model = YOLO(model_arg)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=run_name,
        freeze=args.freeze,
        lr0=args.lr0,
        patience=args.patience,
        cache=args.cache,
        resume=args.resume,
        plots=True,
        save=True,
        exist_ok=True,
        amp=args.amp,
    )

    best = root / args.project / run_name / "weights" / "best.pt"
    last = root / args.project / run_name / "weights" / "last.pt"
    print("\nTraining complete.")
    print(f"Best weights: {best}")
    print(f"Last weights: {last}")
    print("Copy the best.pt into the project root with a clear name, e.g. herring_yolo12n_best.pt, so the GUI can discover it.")


if __name__ == "__main__":
    main()
