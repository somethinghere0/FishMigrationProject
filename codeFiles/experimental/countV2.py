import cv2
import sys
import os

from ultralytics import solutions

# ─── Input / Output ───────────────────────────────────────────────────────────
in_file = sys.argv[1] if len(sys.argv) > 1 else 'processed_videos/barrel_fixed_shortClip_20260217_205047.mp4'
assert os.path.exists(in_file), f"File not found: {in_file}"

cap = cv2.VideoCapture(in_file)
assert cap.isOpened(), "Error reading video file"

output_filename = input("Enter output video filename (without extension): ").strip()
if not output_filename:
    output_filename = "counting_output_v2"
if not output_filename.endswith(('.mp4', '.avi')):
    output_filename += ".mp4"

# ─── Video Properties ─────────────────────────────────────────────────────────
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# ─── Counting Gate ────────────────────────────────────────────────────────────
# Use a narrow RECTANGLE instead of a 2-point line to avoid double-counting
# when a fish's centroid wobbles back and forth across the line.
# The fish must fully enter and exit the gate region to increment the count.

gate_half_width = 40   # Half-width of the gate in pixels (tune as needed)
center_x = w // 2

region_points = [
    (center_x - gate_half_width, 0),     # top-left
    (center_x + gate_half_width, 0),     # top-right
    (center_x + gate_half_width, h),     # bottom-right
    (center_x - gate_half_width, h),     # bottom-left
]

# ─── Video Writer ─────────────────────────────────────────────────────────────
video_writer = cv2.VideoWriter(
    output_filename,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# ─── Object Counter ───────────────────────────────────────────────────────────
counter = solutions.ObjectCounter(
    show=True,
    region=region_points,
    model='current_fish_model.pt',

    # ByteTrack handles lower-confidence detections better than BotSORT,
    # which is important for underwater footage where detections are noisy.
    tracker="bytetrack.yaml",

    # Confidence threshold: lower means more fish detected but more false positives.
    # Raise to 0.5+ if you're getting spurious detections from sand/bubbles.
    conf=0.35,

    # IoU threshold for matching detections to tracks.
    iou=0.5,
)

print(f"\nInput  : {in_file} ({w}x{h} @ {fps:.2f} fps)")
print(f"Output : {output_filename}")
print(f"Gate   : x=[{center_x - gate_half_width}, {center_x + gate_half_width}]\n")

# ─── Process Video ────────────────────────────────────────────────────────────
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_num = 0

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("\nVideo processing complete.")
        break

    results = counter(im0)
    video_writer.write(results.plot_im)

    frame_num += 1
    if frame_num % 30 == 0:
        pct = (frame_num / total * 100) if total > 0 else 0
        print(f"  {frame_num}/{total} frames ({pct:.1f}%)", end="\r")

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# ─── Final Count Summary ──────────────────────────────────────────────────────
try:
    in_count  = counter.in_count
    out_count = counter.out_count
    print(f"\n{'='*40}")
    print(f"  Fish counted IN  : {in_count}")
    print(f"  Fish counted OUT : {out_count}")
    print(f"  Total crossings  : {in_count + out_count}")
    print(f"{'='*40}\n")
except AttributeError:
    print("\nCould not retrieve final count from counter.")
