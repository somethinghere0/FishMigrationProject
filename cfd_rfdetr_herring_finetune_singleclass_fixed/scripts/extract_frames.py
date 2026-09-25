import argparse
from pathlib import Path
import cv2
from tqdm import tqdm


def extract_frames(video_path: Path, out_dir: Path, interval: int, max_frames: int | None, resize_width: int | None):
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {video_path}")
    print(f"Frames={total}, FPS={fps:.2f}, Resolution={w}x{h}")
    print(f"Saving every {interval} frame(s) to {out_dir}")
    saved = 0
    idx = 0
    pbar = tqdm(total=total, desc="Extracting")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            if resize_width is not None and resize_width > 0:
                scale = resize_width / frame.shape[1]
                new_h = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (resize_width, new_h))
            out_path = out_dir / f"{video_path.stem}_frame_{idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    print(f"Saved {saved} frames.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="videos/Herring.mp4", help="Path to input herring video")
    ap.add_argument("--out", default="frames_raw/Herring", help="Output frame directory")
    ap.add_argument("--interval", type=int, default=30, help="Save every Nth frame")
    ap.add_argument("--max-frames", type=int, default=300, help="Maximum frames to save; use 0 for no limit")
    ap.add_argument("--resize-width", type=int, default=0, help="Optional resize width; 0 keeps original size")
    args = ap.parse_args()
    extract_frames(
        Path(args.video),
        Path(args.out),
        max(1, args.interval),
        None if args.max_frames == 0 else args.max_frames,
        None if args.resize_width == 0 else args.resize_width,
    )
