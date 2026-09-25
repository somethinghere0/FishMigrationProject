from __future__ import annotations
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

from .detectors import build_detector
from .preprocess import preprocess_frame_bgr
from .tracker import ByteTrackLite, LineCounter
from .utils import list_inputs, VIDEO_EXTS, IMAGE_EXTS, draw_box_label


def draw_counting_line(frame, line, count):
    x1, y1, x2, y2 = [int(v) for v in line]
    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(frame, f"Count: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 3, cv2.LINE_AA)


def run_image(path: Path, detector, cfg: dict, output_dir: Path):
    img = cv2.imread(str(path))
    if img is None:
        print(f"Could not read image: {path}")
        return []
    proc = preprocess_frame_bgr(img, cfg.get("preprocess", {}))
    det = detector.predict_bgr(proc)
    rows = []
    ann = img.copy()
    for i, (box, score) in enumerate(zip(det.boxes, det.scores)):
        draw_box_label(ann, box, f"fish {score:.2f}", (0, 255, 0), 2)
        x1, y1, x2, y2 = box.tolist()
        rows.append({
            "source": str(path), "frame": 0, "det_id": i, "track_id": None,
            "score": float(score), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "count_so_far": len(det.boxes), "new_count": 0,
        })
    out_img = output_dir / "annotated_images" / path.name
    out_img.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), ann)
    return rows


def run_video(path: Path, detector, cfg: dict, output_dir: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Could not open video: {path}")
        return [], {"source": str(path), "predicted_count": 0, "frames_processed": 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    tracker_cfg = cfg.get("tracking", {})
    tracker = ByteTrackLite(
        high_thresh=tracker_cfg.get("high_thresh", 0.35),
        low_thresh=tracker_cfg.get("low_thresh", 0.10),
        match_iou_thresh=tracker_cfg.get("match_iou_thresh", 0.25),
        max_lost_frames=tracker_cfg.get("max_lost_frames", 30),
    )
    count_cfg = cfg.get("counting", {})
    line = count_cfg.get("line", [w // 2, 0, w // 2, h])
    counter = LineCounter(
        line=line,
        direction=count_cfg.get("direction", "right"),
        min_hits_before_count=tracker_cfg.get("min_hits_before_count", 2),
    )

    save_video = bool(cfg.get("video", {}).get("save_annotated_video", True))
    out_writer = None
    if save_video:
        out_vid_dir = output_dir / "annotated_videos"
        out_vid_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_vid_dir / f"{path.stem}_counted.mp4"
        out_fps = cfg.get("video", {}).get("output_fps", None) or fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(out_path), fourcc, float(out_fps), (w, h))

    save_frames = bool(cfg.get("video", {}).get("save_annotated_frames", False))
    frame_stride = int(cfg.get("video", {}).get("frame_stride", 1))
    rows = []
    frame_idx = -1
    frames_processed = 0

    pbar = tqdm(total=total if total > 0 else None, desc=f"Processing {path.name}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        pbar.update(1)
        if frame_idx % frame_stride != 0:
            continue

        proc = preprocess_frame_bgr(frame, cfg.get("preprocess", {}))
        det = detector.predict_bgr(proc)
        tracks = tracker.update(det.boxes, det.scores)
        new_count = counter.update(tracks)
        frames_processed += 1

        ann = frame.copy()
        if count_cfg.get("draw_line", True):
            draw_counting_line(ann, line, counter.count)

        for t in tracks:
            if t.lost > 0:
                continue
            color = (0, 255, 0) if not t.counted else (0, 165, 255)
            draw_box_label(ann, t.box, f"ID {t.track_id} fish {t.score:.2f}", color, 2)
            x1, y1, x2, y2 = t.box.tolist()
            rows.append({
                "source": str(path), "frame": frame_idx, "det_id": None, "track_id": t.track_id,
                "score": float(t.score), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "count_so_far": counter.count, "new_count": int(t.counted and t.track_id in counter.counted_ids),
            })

        if save_frames:
            frame_dir = output_dir / "annotated_frames" / path.stem
            frame_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frame_dir / f"frame_{frame_idx:06d}.jpg"), ann)
        if out_writer is not None:
            out_writer.write(ann)

    pbar.close()
    cap.release()
    if out_writer is not None:
        out_writer.release()

    summary = {"source": str(path), "predicted_count": counter.count, "frames_processed": frames_processed}
    return rows, summary


def run_pipeline(cfg: dict):
    output_dir = Path(cfg["input"].get("output_dir", "outputs/results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = build_detector(cfg["model"])
    inputs = list_inputs(cfg["input"]["source"])
    if not inputs:
        raise RuntimeError(f"No images/videos found in {cfg['input']['source']}")

    all_rows = []
    summaries = []
    for p in inputs:
        if p.suffix.lower() in IMAGE_EXTS:
            rows = run_image(p, detector, cfg, output_dir)
            all_rows.extend(rows)
            summaries.append({"source": str(p), "predicted_count": len(rows), "frames_processed": 1})
        elif p.suffix.lower() in VIDEO_EXTS:
            rows, summary = run_video(p, detector, cfg, output_dir)
            all_rows.extend(rows)
            summaries.append(summary)

    pd.DataFrame(all_rows).to_csv(output_dir / "detections_tracks.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "video_counts.csv", index=False)
    print(f"\nDone. Results saved to: {output_dir}")
    print(f"- detections/tracks: {output_dir / 'detections_tracks.csv'}")
    print(f"- counts:            {output_dir / 'video_counts.csv'}")
