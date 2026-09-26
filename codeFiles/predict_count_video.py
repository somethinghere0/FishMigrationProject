"""Run the newly trained YOLO model on a herring video with BoT-SORT tracking.
This creates an annotated video using Ultralytics track mode.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import os
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='runs/train/herring_yolo_retrain/weights/best.pt')
    ap.add_argument('--source', default='videos/Herring.mp4')
    ap.add_argument('--tracker', default='botsort.yaml', choices=['botsort.yaml','bytetrack.yaml'])
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--device', default='0')
    ap.add_argument('--project', default='runs/predict')
    ap.add_argument('--name', default='herring_yolo_tracking')
    args = ap.parse_args()
    os.chdir(ROOT)
    model = YOLO(args.model)
    model.track(
        source=args.source,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        persist=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    print(f"Prediction/tracking complete. Check: {ROOT / args.project / args.name}")

if __name__ == '__main__':
    main()
