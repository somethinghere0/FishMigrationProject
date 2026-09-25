from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.ops import nms

from src.botsort_counter import BoTSORTCounter
from src.model import build_detector
from src.preprocessing import preprocess_image
from src.utils import ensure_dir, load_config, resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', default='runs/video_count')
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get('device', 'cuda'))
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    num_classes = 2 if cfg['classes']['mode'] == 'all_fish' else len(cfg['classes']['fish_category_names']) + 1
    model = build_detector(num_classes, cfg['train']['model'], pretrained_backbone=False)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    out_dir = ensure_dir(args.output)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_dir / 'tracked_counted.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    tracker = BoTSORTCounter(cfg['tracking']['iou_match_threshold'], cfg['tracking']['max_age'], cfg['tracking']['min_hits'])
    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        proc = preprocess_image(frame, cfg['preprocessing']) if cfg['preprocessing'].get('enabled', False) else frame
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(Image.fromarray(rgb)).to(device)
        with torch.no_grad():
            out = model([tensor])[0]
        scores = out['scores'].detach().cpu(); boxes = out['boxes'].detach().cpu()
        keep = scores >= cfg['tracking']['detector_score_threshold']
        boxes, scores = boxes[keep], scores[keep]
        if len(boxes):
            keep2 = nms(boxes, scores, cfg['inference']['nms_threshold'])
            boxes, scores = boxes[keep2].numpy(), scores[keep2].numpy()
        else:
            boxes, scores = np.empty((0,4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        active_tracks = tracker.update(boxes, scores)
        unique_count = tracker.count_unique_tracks()
        for t in active_tracks:
            x1,y1,x2,y2 = [int(v) for v in t.bbox]
            cv2.rectangle(proc, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(proc, f'ID {t.track_id}', (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
        cv2.putText(proc, f'Unique fish count: {unique_count}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        # If preprocessing resized frame, resize back for writer.
        if proc.shape[1] != w or proc.shape[0] != h:
            proc = cv2.resize(proc, (w, h))
        writer.write(proc)
        rows.append({'frame': frame_idx, 'detections': len(boxes), 'unique_track_count': unique_count})
        frame_idx += 1
    cap.release(); writer.release()
    pd.DataFrame(rows).to_csv(out_dir / 'video_counts_by_frame.csv', index=False)
    print(f'Final unique fish count: {tracker.count_unique_tracks()}')
    print(f'Wrote {out_dir}')


if __name__ == '__main__':
    main()
