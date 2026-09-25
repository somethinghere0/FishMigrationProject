#!/usr/bin/env bash
set -e
.venv/bin/python scripts/validate_yolo_labels.py --dataset datasets/herring_yolo
