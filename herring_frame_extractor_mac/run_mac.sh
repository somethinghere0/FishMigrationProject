#!/bin/bash
set -e
source .venv/bin/activate
python extract_frames.py --video videos/Herring.mp4 --output extracted_frames/Herring --interval 30
