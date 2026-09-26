#!/usr/bin/env python3
"""
Gamma Correction Video Enhancer

This script applies non-linear brightness adjustment (Gamma Correction) to a video.
Unlike standard linear brightness/contrast, gamma correction can pull detail out 
of dark shadows or mid-tones without blowing out the already bright highlights 
(like a sandy bottom).

Usage:
    python3 enhance_gamma.py [input_video] [output_video]

    Defaults:
        input_video: output_fixed.mp4
        output_video: processed_videos/gamma_corrected_[timestamp].mp4
"""

import cv2
import numpy as np
import sys
import os
import datetime

# --- Tuning Parameters ---
# Gamma value: 
# < 1.0 = Darkens the image (compresses shadows)
# > 1.0 = Lightens the image (expands shadows/mid-tones)
# Try 1.2 to 1.5 to lift dark areas.
GAMMA = 1.3 

def get_output_path(input_path, prefix="gamma_corrected"):
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

def adjust_gamma(image, gamma=1.0):
    """
    Builds a lookup table mapping the pixel values [0, 255] to
    their adjusted gamma values.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

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

    print(f"Processing (Gamma={GAMMA})... Output will be saved to '{output_path}'")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply Gamma Correction
        gamma_corrected = adjust_gamma(frame, gamma=GAMMA)

        out.write(gamma_corrected)

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
        
    out_file = get_output_path(in_file, prefix="gamma_corrected")
    process_video(in_file, out_file)
