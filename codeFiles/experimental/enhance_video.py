#!/usr/bin/env python3
"""
Video Enhancer Tool

This script applies color and exposure corrections to a video.
Specific enhancements:
1. CLAHE Contrast Enhancement (L-channel)
2. Regional Exposure Correction (Top-Right Darkening)

Usage:
    python3 enhance_video.py [input_video] [output_video]

    Defaults:
        input_video: output_fixed.mp4
        output_video: output_enhanced.mp4
"""

import cv2
import numpy as np
import sys
import os

def process_video(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{input_path}'.")
        return

    # Video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input: {input_path} ({w}x{h} @ {fps:.2f} fps)")

    # Output setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Enhancement setup: CLAHE
    # ClipLimit 2.0 is moderate contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # Enhancement setup: Exposure Mask (Gradient)
    # Define Top-Right gradient region
    center_x = w
    center_y = 0
    radius = min(h, w) * 0.5 # Adjustment radius
    
    # Precompute mask
    y_grid, x_grid = np.indices((h, w))
    dist = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    
    # Calculate influence factor (0 to 1)
    influence = np.clip(1 - (dist / radius), 0, 1)
    
    # Adjustment factor: 1.0 (normal) -> 0.6 (darker)
    adjustment_factor = 1.0 - (influence * 0.4)
    # Expand for 3 channels
    adjustment_factor = adjustment_factor[:, :, np.newaxis]

    print(f"Processing... Output will be saved to '{output_path}'")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Contrast Enhancement (CLAHE on L-channel)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 2. Exposure Correction (Top-Right)
        # Convert to float for multiplication, then back to uint8
        final = (enhanced.astype(np.float32) * adjustment_factor).astype(np.uint8)

        out.write(final)

        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Processed {frame_count} frames ({progress:.1f}%)", end='\r')

    print(f"\nDone! Processed {frame_count} frames.")
    cap.release()
    out.release()

import datetime

def get_output_path(input_path, prefix="enhanced"):
    """
    Generates a timestamped output path in 'processed_videos/' directory.
    """
    # Create directory if it doesn't exist
    output_dir = "processed_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    
    # Avoid double prefixing if input already has prefix
    if name.startswith("barrel_fixed_"):
        name = name.replace("barrel_fixed_", "")
    
    new_filename = f"{prefix}_{name}_{timestamp}{ext}"
    return os.path.join(output_dir, new_filename)

if __name__ == "__main__":
    # Handle arguments
    in_file = "output_fixed.mp4"

    if len(sys.argv) > 1:
        in_file = sys.argv[1]
        
    # Auto-generate output path
    out_file = get_output_path(in_file, prefix="enhanced")

    process_video(in_file, out_file)
