import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int = 30,
    image_ext: str = "jpg",
    resize_width: int | None = None,
) -> None:
    """Extract frames from a video and save them as images."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found: {video_path}\n"
            "Place your video inside the videos/ folder or pass --video with the correct path."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps and fps > 0 else 0

    print("Video information")
    print("-----------------")
    print(f"Video path: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Resolution: {width} x {height}")
    print(f"Duration: {duration_sec:.2f} seconds")
    print(f"Saving every {frame_interval} frame(s)")
    print(f"Output folder: {output_dir}")
    print()

    saved_count = 0
    frame_idx = 0

    with tqdm(total=total_frames if total_frames > 0 else None, desc="Extracting frames") as progress_bar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                if resize_width is not None:
                    scale = resize_width / frame.shape[1]
                    new_height = int(frame.shape[0] * scale)
                    frame = cv2.resize(frame, (resize_width, new_height))

                output_name = f"{video_path.stem}_frame_{frame_idx:06d}.{image_ext}"
                output_path = output_dir / output_name
                success = cv2.imwrite(str(output_path), frame)
                if not success:
                    raise RuntimeError(f"Could not save frame to: {output_path}")
                saved_count += 1

            frame_idx += 1
            progress_bar.update(1)

    cap.release()

    print()
    print("Extraction complete")
    print("-------------------")
    print(f"Frames read: {frame_idx}")
    print(f"Frames saved: {saved_count}")
    print(f"Saved to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from herring video.")
    parser.add_argument(
        "--video",
        default="videos/Herring.mp4",
        help="Path to input video. Default: videos/Herring.mp4",
    )
    parser.add_argument(
        "--output",
        default="extracted_frames/Herring",
        help="Output folder for extracted frames. Default: extracted_frames/Herring",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Save one frame every N frames. 30 is about 1 FPS for 30 FPS video.",
    )
    parser.add_argument(
        "--ext",
        choices=["jpg", "png"],
        default="jpg",
        help="Output image format. Default: jpg",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Optional resize width, preserving aspect ratio. Example: --resize-width 640",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        frame_interval=args.interval,
        image_ext=args.ext,
        resize_width=args.resize_width,
    )
