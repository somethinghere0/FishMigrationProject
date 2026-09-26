#!/usr/bin/env python3
"""
HDR Tone Mapping Video Enhancer

Uses OpenCV's built-in Reinhard tone mapping to simulate HDR processing on a 
standard video. This compresses blown-out highlights (bright sandy bottom, sun glare)
while simultaneously boosting mid-tone contrast — making fish stand out more clearly
against the background without losing quality.

Unlike simple brightness reduction, tone mapping operates in floating-point HDR space
so transitions between bright and dark areas remain smooth and natural.

Usage:
    python3 fix_hdr.py <input_video>

    Output: processed_videos/hdr_<name>_<timestamp>.mp4

Tunable Parameters (edit below):
    GAMMA        - Controls overall brightness. < 1.0 = brighter, > 1.0 = darker.
                   Try 1.0–1.5. Default: 1.2 (slight darkening of highlights).
    SATURATION   - Color saturation after tone mapping. 1.0 = natural, 1.5 = vivid.
                   Boosting helps fish pop against sand. Default: 1.3.
    INTENSITY    - Reinhard intensity. Higher = more compression of highlights.
                   Range: -8 to 8. Default: 0.0 (neutral).
    LIGHT_ADAPT  - How much the tone map adapts to local vs global lighting.
                   0.0 = global, 1.0 = fully local. Default: 0.5.
    COLOR_ADAPT  - Per-channel vs luminance adaptation. 0.0 = luminance, 1.0 = per-channel.
                   Default: 0.0.
    CLAHE_AFTER  - Whether to apply CLAHE contrast enhancement after tone mapping.
                   Helps fish edges stand out for YOLO. Default: True.
"""

import cv2
import numpy as np
import sys
import os
import datetime

# ─── Tunable Parameters ───────────────────────────────────────────────────────
GAMMA        = 1.2    # Overall brightness curve (1.0 = neutral)
SATURATION   = 1.3    # Color saturation boost
INTENSITY    = 0.0    # Reinhard highlight compression
LIGHT_ADAPT  = 0.5    # Local vs global light adaptation
COLOR_ADAPT  = 0.0    # Per-channel vs luminance adaptation
CLAHE_AFTER  = True   # Apply CLAHE after tone mapping for extra fish contrast
# ──────────────────────────────────────────────────────────────────────────────

def get_output_path(input_path):
    output_dir = "processed_videos"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = os.path.splitext(os.path.basename(input_path))[0]
    for prefix in ("barrel_fixed_", "exposure_fixed_", "color_balanced_", "enhanced_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return os.path.join(output_dir, f"hdr_{name}_{timestamp}.mp4")

def apply_tone_map(frame, tonemap):
    """
    Applies Reinhard HDR tone mapping to a single BGR frame.
    1. Convert to float32 HDR (0.0–1.0 range)
    2. Apply tone map (compresses highlights, lifts shadows)
    3. Convert back to uint8
    """
    # Convert to float32 in [0, 1] range
    hdr = frame.astype(np.float32) / 255.0

    # Apply Reinhard tone mapping
    ldr = tonemap.process(hdr)

    # Clip and scale back to [0, 255]
    ldr = np.clip(ldr * 255.0, 0, 255).astype(np.uint8)
    return ldr

def boost_saturation(frame, saturation_scale):
    """Boosts color saturation in HSV space."""
    if abs(saturation_scale - 1.0) < 0.01:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_clahe(frame):
    """Applies CLAHE to the L channel to boost local contrast."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

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
    print(f"Params: gamma={GAMMA}, saturation={SATURATION}, intensity={INTENSITY}")
    print(f"        light_adapt={LIGHT_ADAPT}, color_adapt={COLOR_ADAPT}, clahe={CLAHE_AFTER}")
    print(f"Output: {output_path}\n")

    # Build the Reinhard tone mapper once (reused every frame)
    tonemap = cv2.createTonemapReinhard(
        gamma=GAMMA,
        intensity=INTENSITY,
        light_adapt=LIGHT_ADAPT,
        color_adapt=COLOR_ADAPT
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: HDR tone mapping
        result = apply_tone_map(frame, tonemap)

        # Step 2: Saturation boost (makes fish colors pop)
        result = boost_saturation(result, SATURATION)

        # Step 3: Optional CLAHE for local contrast (helps YOLO detect edges)
        if CLAHE_AFTER:
            result = apply_clahe(result)

        writer.write(result)
        count += 1
        if count % 30 == 0:
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {count}/{total} frames ({pct:.1f}%)", end="\r")

    cap.release()
    writer.release()
    print(f"\nDone – saved to '{output_path}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 fix_hdr.py <input_video> [output_video]")
        sys.exit(1)

    in_file = sys.argv[1]
    if not os.path.exists(in_file):
        print(f"Error: File '{in_file}' not found.")
        sys.exit(1)

    # Optional second argument lets callers (e.g. the GUI) specify the output path.
    if len(sys.argv) >= 3:
        out_file = sys.argv[2]
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    else:
        out_file = get_output_path(in_file)

    process_video(in_file, out_file)
