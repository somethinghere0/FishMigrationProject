#!/usr/bin/env python3
"""
Exposure Correction Video Enhancer

This script applies an adaptive highlight mask to correct overexposed areas
(like a bright sandy bottom reflections) without crushing the shadows where 
fish might be hiding.

Usage:
    python3 enhance_exposure.py [input_video] [output_video]

    Defaults:
        input_video: output_fixed.mp4
        output_video: processed_videos/exposure_fixed_[timestamp].mp4
"""

import cv2
import numpy as np
import sys
import os
import datetime

# --- Tuning Parameters ---
# Use tune_exposure.py to find the best values for your video
THRESHOLD = 200    # Pixel intensity to consider "overexposed" (0-255)
REDUCTION = 50     # Percentage to reduce brightness (0-100%)
BLUR_SIZE = 21     # Kernel size for blurring the mask (must be odd)

def get_output_path(input_path, prefix="exposure_fixed"):
    """Generates a timestamped output path in 'processed_videos/' directory."""
    output_dir = "processed_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    
    # Avoid double prefixing
    if name.startswith("barrel_fixed_"):
        name = name.replace("barrel_fixed_", "")
    
    new_filename = f"{prefix}_{name}_{timestamp}{ext}"
    return os.path.join(output_dir, new_filename)

def correct_exposure(image, threshold=200, reduction_percent=50, blur_size=21):
    """
    Isolates highlights using a threshold, blurs the mask for a smooth transition,
    and applies a linear brightness reduction ONLY to the masked (bright) areas.
    """
    image_f = image.astype(np.float32)

    # 1. Convert to grayscale to find bright spots
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Create mask of overexposed areas
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # 3. Blur the mask for smooth blending
    blurred_mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    
    # Normalize mask to 0.0 - 1.0 range
    mask_norm = blurred_mask.astype(np.float32) / 255.0
    mask_norm_3c = cv2.merge([mask_norm, mask_norm, mask_norm])
    
    # 4. Create the darkened version
    # Calculate fraction remaining (e.g., 50% reduction = multiply by 0.5)
    fraction = 1.0 - (reduction_percent / 100.0)
    darkened = image_f * fraction
    
    # 5. Blend: Original where mask is 0 (dark), Darkened where mask is 1 (bright)
    blended = image_f * (1.0 - mask_norm_3c) + darkened * mask_norm_3c
    return np.clip(blended, 0, 255).astype(np.uint8)

def process_video(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{input_path}'.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input: {input_path} ({w}x{h} @ {fps:.2f} fps)")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"Processing... Output will be saved to '{output_path}'")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corrected = correct_exposure(frame, threshold=THRESHOLD, gamma=GAMMA, blur_size=BLUR_SIZE)

        out.write(corrected)

        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Processed {frame_count} frames ({progress:.1f}%)", end='\r')

    print(f"\nDone! Processed {frame_count} frames.")
    cap.release()
    out.release()

if __name__ == "__main__":
    in_file = "output_fixed.mp4"

    if len(sys.argv) > 1:
        in_file = sys.argv[1]
        
    out_file = get_output_path(in_file, prefix="exposure_fixed")
    process_video(in_file, out_file)
