#!/usr/bin/env bash
set -e
source .venv/bin/activate
python scripts/prepare_yolo_dataset.py --images frames_labeled/images --labels frames_labeled/labels --out datasets/herring_yolo
