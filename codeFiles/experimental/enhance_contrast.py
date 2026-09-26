#!/usr/bin/env python3
"""
Fish Contrast Enhancer (YOLO-optimized)

Designed specifically to improve YOLO fish detection accuracy by boosting the
visual contrast of fish against the underwater background.

Strategy:
  1. Background subtraction (MOG2) to detect which pixels are moving (i.e., fish).
  2. Dilate the foreground mask so the full fish body (not just edge pixels) is captured.
  3. Apply CLAHE and a brightness boost ONLY inside the foreground mask.
  4. Apply a gentle global CLAHE to the rest of the frame so the background
     doesn't become noisy and generate false YOLO detections.
  5. Blend the result back together.

This means fish get a strong contrast boost while the sandy bottom stays calm,
which is exactly the signal-to-noise ratio improvement YOLO needs.

Usage:
    python3 enhance_contrast.py <input_video>

    Output: processed_videos/contrast_<name>_<timestamp>.mp4

Tunable Parameters (edit below):
    FG_CLIP_LIMIT      - CLAHE clip limit for fish regions. Higher = more contrast.
    BG_CLIP_LIMIT      - CLAHE clip limit for background.  Lower = stays calm.
    FG_BRIGHTNESS      - Additive brightness boost for fish regions (0–50).
    MOG2_HISTORY       - How many frames MOG2 uses to build the background model.
                         Lower = adapts faster to new backgrounds, but noisier mask.
    MOG2_THRESHOLD     - Sensitivity of foreground detection. Lower = more sensitive.
    MASK_DILATE_PX     - How many pixels to expand the foreground mask around each fish.
                         Ensures the full fish body is covered, not just the edges.
"""

import cv2
import numpy as np
import sys
import os
import datetime

# ─── Tunable Parameters ───────────────────────────────────────────────────────
FG_CLIP_LIMIT   = 4.0   # CLAHE strength on fish (higher = more contrast)
BG_CLIP_LIMIT   = 1.5   # CLAHE strength on background (keep low to avoid noise)
FG_BRIGHTNESS   = 25    # Additive brightness boost inside fish mask (0-100)
MOG2_HISTORY    = 100   # Frames used to build background model
MOG2_THRESHOLD  = 40    # Foreground detection sensitivity (lower = more sensitive)
MASK_DILATE_PX  = 20    # Pixels to expand fish mask (covers full fish body)
# ──────────────────────────────────────────────────────────────────────────────

def get_output_path(input_path):
    output_dir = "processed_videos"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = os.path.splitext(os.path.basename(input_path))[0]
    for prefix in ("barrel_fixed_", "exposure_fixed_", "hdr_", "color_balanced_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return os.path.join(output_dir, f"contrast_{name}_{timestamp}.mp4")

def apply_clahe(frame, clip_limit=2.0, tile_size=8):
    """Apply CLAHE to the L channel of a BGR frame."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

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
    print(f"Output: {output_path}\n")

    # Background subtractor — learns the sandy bottom over time
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_THRESHOLD,
        detectShadows=False   # Shadows would add noise; we only want fish
    )

    # Dilation kernel to expand fish mask to cover full body
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MASK_DILATE_PX * 2 + 1, MASK_DILATE_PX * 2 + 1)
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Step 1: Get foreground mask for this frame ──────────────────────
        raw_mask = subtractor.apply(frame)  # 0=background, 255=foreground

        # Expand the mask to cover full fish body (not just moving edges)
        fish_mask = cv2.dilate(raw_mask, dilate_kernel)

        # Convert to float mask (0.0–1.0) for blending
        mask_f = fish_mask.astype(np.float32) / 255.0
        mask_3c = cv2.merge([mask_f, mask_f, mask_f])

        # ── Step 2: Apply strong CLAHE to fish regions ──────────────────────
        fg_enhanced = apply_clahe(frame, clip_limit=FG_CLIP_LIMIT, tile_size=8)

        # Add a brightness boost inside the fish mask
        if FG_BRIGHTNESS > 0:
            boost = np.full_like(fg_enhanced, FG_BRIGHTNESS, dtype=np.uint8)
            fg_enhanced = cv2.add(fg_enhanced, boost)

        # ── Step 3: Apply gentle CLAHE to background regions ─────────────────
        bg_enhanced = apply_clahe(frame, clip_limit=BG_CLIP_LIMIT, tile_size=16)

        # ── Step 4: Blend based on mask ──────────────────────────────────────
        fg_f = fg_enhanced.astype(np.float32)
        bg_f = bg_enhanced.astype(np.float32)
        blended = fg_f * mask_3c + bg_f * (1.0 - mask_3c)
        result = np.clip(blended, 0, 255).astype(np.uint8)

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
        print("Usage: python3 enhance_contrast.py <input_video>")
        sys.exit(1)

    in_file = sys.argv[1]
    if not os.path.exists(in_file):
        print(f"Error: File '{in_file}' not found.")
        sys.exit(1)

    out_file = get_output_path(in_file)
    process_video(in_file, out_file)
