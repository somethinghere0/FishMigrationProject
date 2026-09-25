#!/usr/bin/env bash
set -e
source .venv/bin/activate
python scripts/predict_video_rfdetr.py --video videos/Herring.mp4 --checkpoint outputs/cfd_herring_finetune/checkpoint_best_total.pth --model-size nano --out-video outputs/Herring_finetuned_prediction.mp4 --out-csv outputs/Herring_finetuned_detections.csv
