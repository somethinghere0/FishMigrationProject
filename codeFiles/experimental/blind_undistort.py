import cv2
import numpy as np

# ==========================
# Load image
# ==========================
img = cv2.imread("preview.jpg")
if img is None:
    raise FileNotFoundError("input.jpg not found")

h, w = img.shape[:2]

# ==========================
# Detect circular fisheye mask
# ==========================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9, 9), 0)

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

# ==========================
# Fisheye reprojection
# ==========================
out_w, out_h = w, h
fov = np.deg2rad(150)  # strong fisheye

map_x = np.zeros((out_h, out_w), np.float32)
map_y = np.zeros((out_h, out_w), np.float32)

for y in range(out_h):
    for x in range(out_w):
        nx = (x - out_w / 2) / (out_w / 2)
        ny = (y - out_h / 2) / (out_h / 2)

        r_xy = np.sqrt(nx * nx + ny * ny)
        if r_xy > 1:
            map_x[y, x] = -1
            map_y[y, x] = -1
            continue

        theta = r_xy * fov / 2
        phi = np.arctan2(ny, nx)

        src_r = theta / (fov / 2) * r
        src_x = cx + src_r * np.cos(phi)
        src_y = cy + src_r * np.sin(phi)

        map_x[y, x] = src_x
        map_y[y, x] = src_y

rectilinear = cv2.remap(
    img,
    map_x,
    map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT
)

cv2.imwrite("output_rectilinear.jpg", rectilinear)
print("Saved output_rectilinear.jpg")
