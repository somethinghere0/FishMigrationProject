import cv2
import numpy as np

# ==========================
# Configuration
# ==========================
INPUT_VIDEO = "testVideos/shortClip.mp4"
OUTPUT_VIDEO = "output_rectilinear.mp4"

FOV_DEG = 160          # fisheye field of view (tune this)
BLUR_KERNEL = (9, 9)   # for circle detection

# ==========================
# Open video
# ==========================
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame")

h, w = first_frame.shape[:2]

# ==========================
# Detect fisheye circle (once)
# ==========================
gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)

circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=h,
    param1=100,
    param2=30,
    minRadius=int(0.4 * min(w, h)),
    maxRadius=int(0.55 * min(w, h))
)

if circles is None:
    raise RuntimeError("Could not detect fisheye circle")

cx, cy, r = np.round(circles[0][0]).astype(int)
print(f"Detected fisheye circle: center=({cx},{cy}), radius={r}")

# ==========================
# Precompute remap grids
# ==========================
fov = np.deg2rad(FOV_DEG)

map_x = np.full((h, w), -1, dtype=np.float32)
map_y = np.full((h, w), -1, dtype=np.float32)

for y in range(h):
    for x in range(w):
        nx = (x - w / 2) / (w / 2)
        ny = (y - h / 2) / (h / 2)

        r_xy = np.sqrt(nx * nx + ny * ny)
        if r_xy > 1:
            continue

        theta = r_xy * fov / 2
        phi = np.arctan2(ny, nx)

        src_r = theta / (fov / 2) * r
        src_x = cx + src_r * np.cos(phi)
        src_y = cy + src_r * np.sin(phi)

        map_x[y, x] = src_x
        map_y[y, x] = src_y

# ==========================
# Video writer
# ==========================
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

# ==========================
# Process frames
# ==========================
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rectilinear = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    writer.write(rectilinear)

cap.release()
writer.release()

print(f"Saved rectilinear video to {OUTPUT_VIDEO}")
