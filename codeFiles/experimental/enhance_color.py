#!/usr/bin/env python3
"""
Color Balancing Video Enhancer

This script applies a White Balance correction (Gray World assumption) to a video.
This helps remove the strong blue/green underwater tint, restoring natural colors
that YOLO models are typically trained on.

Usage:
    python3 enhance_color.py [input_video] [output_video]

    Defaults:
        input_video: output_fixed.mp4
        output_video: processed_videos/color_balanced_[timestamp].mp4
"""

import cv2
import numpy as np
import sys
import os
import datetime

def get_output_path(input_path, prefix="color_balanced"):
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

    # We use a white balancer object from OpenCV xphoto module if available,
    # otherwise we implement a fast, simple Gray World approximation.
    
    try:
        # cv2.xphoto might not be in standard opencv-python, requires opencv-contrib-python
        wb = cv2.xphoto.createSimpleWB()
    except AttributeError:
        wb = None
        print("Note: cv2.xphoto not found. Using custom Gray World algorithm.")

    print(f"Processing... Output will be saved to '{output_path}'")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if wb is not None:
            balanced = wb.balanceWhite(frame)
        else:
            # Custom Gray World algorithm approximation
            # Calculate the average of each channel
            avg_b = np.mean(frame[:, :, 0])
            avg_g = np.mean(frame[:, :, 1])
            avg_r = np.mean(frame[:, :, 2])
            
            # The "gray" value is the average of the averages
            avg_gray = (avg_b + avg_g + avg_r) / 3

            # Calculate the gains for each channel
            # We add a small epsilon to avoid division by zero
            eps = 1e-6
            gain_b = avg_gray / (avg_b + eps)
            gain_g = avg_gray / (avg_g + eps)
            gain_r = avg_gray / (avg_r + eps)
            
            # Limit the gains to prevent extreme distortion if one channel is completely missing
            gain_b = min(gain_b, 3.0)
            gain_g = min(gain_g, 3.0)
            gain_r = min(gain_r, 3.0)

            # Apply gains and clip
            balanced = frame.copy().astype(np.float32)
            balanced[:, :, 0] *= gain_b
            balanced[:, :, 1] *= gain_g
            balanced[:, :, 2] *= gain_r
            
            balanced = np.clip(balanced, 0, 255).astype(np.uint8)

        out.write(balanced)

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
        
    out_file = get_output_path(in_file, prefix="color_balanced")
    process_video(in_file, out_file)
