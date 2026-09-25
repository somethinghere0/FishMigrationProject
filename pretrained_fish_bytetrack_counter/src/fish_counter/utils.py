from __future__ import annotations
from pathlib import Path
import yaml
import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_inputs(source: str | Path) -> list[Path]:
    source = Path(source)
    if source.is_file():
        return [source]
    if not source.exists():
        raise FileNotFoundError(f"Input source not found: {source}")
    files = []
    for p in source.rglob("*"):
        if p.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS:
            files.append(p)
    return sorted(files)


def xyxy_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a[N,4], b[M,4]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-6
    return inter / union


def center_xy(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def draw_box_label(img, box, label, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    y_text = max(15, y1 - 5)
    cv2.putText(img, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
