import cv2
import sys
import os

from ultralytics import solutions

import argparse

parser = argparse.ArgumentParser(description="Fish counter")
parser.add_argument("input", nargs="?",
                    default="processed_videos/barrel_fixed_shortClip_20260217_205047.mp4",
                    help="Input video file")
parser.add_argument("--output", "-o", default=None,
                    help="Output video filename (with or without .mp4 extension)")
parser.add_argument("--no-show", action="store_true",
                    help="Suppress the live OpenCV display window")
parser.add_argument("--gui-stream", action="store_true",
                    help="Stream annotated frames as length-prefixed JPEGs to stdout")
parser.add_argument("--conf", type=float, default=0.25,
                    help="Detection confidence threshold (0–1, default 0.25)")
parser.add_argument("--model", default="current_fish_model.pt",
                    help="Path to YOLO .pt model file (default current_fish_model.pt)")
args = parser.parse_args()

# Suppress ultralytics console output when streaming to avoid corrupting binary stdout
if args.gui_stream:
    os.environ.setdefault("YOLO_VERBOSE", "False")
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

in_file = args.input
assert os.path.exists(in_file), f"File not found: {in_file}"
cap = cv2.VideoCapture(in_file)
assert cap.isOpened(), "Error reading video file"

# Determine output filename
if args.output:
    output_filename = args.output
    if not output_filename.endswith((".mp4", ".avi")):
        output_filename += ".mp4"
else:
    output_filename = input("Enter output video filename (without extension): ").strip()
    if not output_filename:
        output_filename = "object_counting_output"
    if not output_filename.endswith((".mp4", ".avi")):
        output_filename += ".mp4"


# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Calculate vertical box in center of video
box_width = 10  # width of the vertical box in pixels
box_top = 0  # start 10% from top
box_bottom = int(h)  # end 10% from bottom
center_x = w // 2
left_x = center_x - box_width // 2 - 100

# Vertical rectangular region (clockwise: top-left, top-right, bottom-right, bottom-left)
region_points = [
    (left_x, box_top),      # top-left
    (left_x, box_bottom)    # bottom-left
]

# region_points = [(20, 400), (1080, 400)]                                      # line counting
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # rectangular region
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon region
video_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize object counter object
counter = solutions.ObjectCounter(
    show=not args.no_show,  # suppressed when called with --no-show
    region=region_points,  # pass region points
    model=args.model,
    tracker="botsort.yaml",  # choose trackers i.e "bytetrack.yaml"
    conf=args.conf,
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = counter(im0)
    processed = results.plot_im

    video_writer.write(processed)  # write the processed frame.

    # Stream annotated frame to stdout for GUI live display
    if args.gui_stream:
        ok, jpeg = cv2.imencode(".jpg", processed,
                                [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            data = jpeg.tobytes()
            sys.stdout.buffer.write(len(data).to_bytes(4, "big"))
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows

# Emit final count to stderr so the GUI can parse it
sys.stderr.write(f"FINAL_IN_COUNT:{counter.in_count}\n")
sys.stderr.write(f"FINAL_OUT_COUNT:{counter.out_count}\n")
sys.stderr.flush()