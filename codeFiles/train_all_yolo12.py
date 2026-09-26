"""Train YOLO12 n/s/m/x variants on the same SeniorProject fish dataset.

This is optional. It is useful when the professor wants to compare speed/accuracy
tradeoffs across YOLO12 model sizes. It calls train_yolo_fish.py repeatedly.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12x.pt"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, help="Model list to train")
    ap.add_argument("--data", default="fish_data.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--freeze", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    # Conservative default batches by size. Students can edit these if GPU memory allows.
    batch_by_model = {
        "yolo12n.pt": 8,
        "yolo12s.pt": 6,
        "yolo12m.pt": 4,
        "yolo12x.pt": 2,
    }

    for model in args.models:
        stem = Path(model).stem
        batch = batch_by_model.get(model, 4)
        name = f"herring_{stem}_retrain"
        cmd = [
            sys.executable,
            str(ROOT / "codeFiles" / "train_yolo_fish.py"),
            "--model", model,
            "--data", args.data,
            "--epochs", str(args.epochs),
            "--imgsz", str(args.imgsz),
            "--batch", str(batch),
            "--device", args.device,
            "--freeze", str(args.freeze),
            "--workers", str(args.workers),
            "--lr0", str(args.lr0),
            "--patience", str(args.patience),
            "--name", name,
        ]
        print("\n" + "=" * 90)
        print("Training", model, "with batch", batch)
        print(" ".join(cmd))
        print("=" * 90)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
