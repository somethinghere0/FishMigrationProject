#!/usr/bin/env bash
set -e
source .venv/bin/activate
python scripts/train_cfd_rfdetr.py --config configs/train_config.yaml
