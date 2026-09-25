#!/usr/bin/env bash
set -e
source .venv/bin/activate
python scripts/extract_frames.py --video videos/Herring.mp4 --out frames_raw/Herring --interval 30 --max-frames 300
