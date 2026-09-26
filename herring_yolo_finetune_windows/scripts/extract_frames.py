import argparse
from pathlib import Path
import cv2
from tqdm import tqdm


def extract_frames(video_path: Path, output_dir: Path, interval: int, max_frames: int | None, prefix: str, resize_width: int | None):
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {video_path}")
    print(f"Frames: {total} | FPS: {fps:.2f} | Resolution: {w}x{h}")
    print(f"Saving every {interval} frame(s) to {output_dir}")

    saved = 0
    frame_idx = 0
    with tqdm(total=total, desc="Extracting") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % interval == 0:
                if resize_width is not None:
                    scale = resize_width / frame.shape[1]
                    new_h = int(frame.shape[0] * scale)
                    frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)
                out_name = f"{prefix}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(output_dir / out_name), frame)
                saved += 1
                if max_frames is not None and saved >= max_frames:
                    break
            frame_idx += 1
            pbar.update(1)
    cap.release()
    print(f"Saved {saved} frame(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract training/annotation frames from herring videos.")
    parser.add_argument("--video", default="videos/Herring.mp4", help="Path to input video")
    parser.add_argument("--output", default="frames_to_label", help="Folder to save frames for annotation")
    parser.add_argument("--interval", type=int, default=30, help="Save one frame every N frames")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum frames to save; use -1 for no limit")
    parser.add_argument("--prefix", default="herring", help="Filename prefix")
    parser.add_argument("--resize-width", type=int, default=-1, help="Resize output width; -1 keeps original")
    args = parser.parse_args()

    extract_frames(
        video_path=Path(args.video),
        output_dir=Path(args.output),
        interval=max(args.interval, 1),
        max_frames=None if args.max_frames < 0 else args.max_frames,
        prefix=args.prefix,
        resize_width=None if args.resize_width < 0 else args.resize_width,
    )
