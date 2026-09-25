from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.ops import nms

from src.model import build_detector
from src.preprocessing import preprocess_image
from src.utils import ensure_dir, load_config, resolve_device


def draw_boxes(image_bgr, boxes, scores, labels):
    out = image_bgr.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f'fish {score:.2f}', (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True, help='Image path or folder of images')
    parser.add_argument('--output', default='runs/inference_images')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get('device', 'cuda'))
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    num_classes = 2 if cfg['classes']['mode'] == 'all_fish' else len(cfg['classes']['fish_category_names']) + 1
    model = build_detector(num_classes, cfg['train']['model'], pretrained_backbone=False)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    inp = Path(args.input)
    exts = set(cfg['train']['image_extensions'])
    files = [inp] if inp.is_file() else [p for p in inp.rglob('*') if p.suffix.lower() in exts]
    out_dir = ensure_dir(args.output)
    rows = []
    for p in files:
        image_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        proc = preprocess_image(image_bgr, cfg['preprocessing']) if cfg['preprocessing'].get('enabled', False) else image_bgr
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(Image.fromarray(rgb)).to(device)
        with torch.no_grad():
            out = model([tensor])[0]
        scores = out['scores'].detach().cpu()
        boxes = out['boxes'].detach().cpu()
        labels = out['labels'].detach().cpu()
        keep = scores >= cfg['inference']['score_threshold']
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if len(boxes) > 0:
            keep_nms = nms(boxes, scores, cfg['inference']['nms_threshold'])
            boxes, scores, labels = boxes[keep_nms], scores[keep_nms], labels[keep_nms]
        count = int(len(boxes))
        rows.append({'image': str(p), 'fish_count': count, 'mean_confidence': float(scores.mean()) if len(scores) else 0.0})
        if cfg['inference'].get('save_annotated', True):
            annotated = draw_boxes(proc, boxes.numpy(), scores.numpy(), labels.numpy())
            cv2.putText(annotated, f'Count: {count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            cv2.imwrite(str(out_dir / f'{p.stem}_counted.jpg'), annotated)
    pd.DataFrame(rows).to_csv(out_dir / 'image_counts.csv', index=False)
    print(f'Wrote {out_dir / "image_counts.csv"}')


if __name__ == '__main__':
    main()
