#!/usr/bin/env python3
"""
Overexposure Correction Tool

Corrects overexposed areas (e.g. bright sandy bottom, sun glare) in a video
by identifying pixels above a brightness threshold and linearly scaling them down.
Non-overexposed areas (like the fish and shadows) are NOT affected.

Usage:
    python3 fix_exposure.py <input_video>

    Output will be saved to: processed_videos/exposure_fixed_<name>_<timestamp>.mp4

Tunable Parameters (edit below):
    HIGHLIGHT_THRESHOLD  - Brightness (0-255) above which a pixel is "overexposed"
                           Lower = affect more of the image. Start around 200.
    DARKEN_AMOUNT        - How much to reduce overexposed areas (0.0 to 1.0)
                           0.5 = cut brightness in half. 0.0 = pitch black. Start around 0.5.
    MASK_BLUR            - How smoothly the correction blends at the edges (odd number).
                           Higher = softer, more gradual transition. Start around 21.
"""

import cv2
import numpy as np
import sys
import os
import datetime

# ─── Tunable Parameters ───────────────────────────────────────────────────────
HIGHLIGHT_THRESHOLD = 210   # 0-255, lower = more of the image gets darkened
DARKEN_AMOUNT       = 0.7   # 0.0-1.0, fraction to multiply bright pixels by (0.7 = 30% darker)
MASK_BLUR           = 21    # odd number; higher = softer blend edges
# ──────────────────────────────────────────────────────────────────────────────

def get_output_path(input_path):
    output_dir = "processed_videos"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = os.path.splitext(os.path.basename(input_path))[0]
    # strip common prefixes to keep filenames readable
    for prefix in ("barrel_fixed_", "color_balanced_", "enhanced_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return os.path.join(output_dir, f"exposure_fixed_{name}_{timestamp}.mp4")

def build_mask(gray_frame, threshold, blur_size):
    """Returns a float32 mask (0-1) of overexposed regions, smoothed at the edges."""
    _, mask = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (blur_size, blur_size), 0)
    return blurred / 255.0   # shape: (H, W), values 0.0–1.0

def fix_frame(frame, threshold, darken_amount, blur_size):
    """Apply highlight reduction to a single frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = build_mask(gray, threshold, blur_size)          # (H, W) float32
    mask3 = mask[:, :, np.newaxis]                         # broadcast over channels

    frame_f = frame.astype(np.float32)
    darkened = frame_f * darken_amount                     # reduce brightness
    blended  = frame_f * (1.0 - mask3) + darkened * mask3 # blend by mask
    return np.clip(blended, 0, 255).astype(np.uint8)

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open '{input_path}'")
        sys.exit(1)

    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input : {input_path}  ({w}x{h} @ {fps:.2f} fps, {total} frames)")
    print(f"Params: threshold={HIGHLIGHT_THRESHOLD}, darken={DARKEN_AMOUNT}, blur={MASK_BLUR}")
    print(f"Output: {output_path}\n")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(fix_frame(frame, HIGHLIGHT_THRESHOLD, DARKEN_AMOUNT, MASK_BLUR))
        count += 1
        if count % 30 == 0:
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {count}/{total} frames ({pct:.1f}%)", end="\r")

    cap.release()
    writer.release()
    print(f"\nDone – saved to '{output_path}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 fix_exposure.py <input_video>")
        sys.exit(1)

    in_file = sys.argv[1]
    if not os.path.exists(in_file):
        print(f"Error: File '{in_file}' not found.")
        sys.exit(1)

    out_file = get_output_path(in_file)
    process_video(in_file, out_file)
