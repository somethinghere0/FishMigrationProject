from __future__ import annotations
import argparse
from pathlib import Path
import os
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='runs/train/herring_yolo_retrain/weights/best.pt')
    ap.add_argument('--data', default='fish_data.yaml')
    ap.add_argument('--device', default='0')
    ap.add_argument('--imgsz', type=int, default=640)
    args = ap.parse_args()
    os.chdir(ROOT)
    model = YOLO(args.model)
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device, plots=True)
    print(metrics)

if __name__ == '__main__':
    main()
