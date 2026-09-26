#!/usr/bin/env python3
"""
Sharpening Video Enhancer

This script applies an Unsharp Mask to the video to improve edge definition without 
overly boosting noise. The YOLO model relies on clear edges to draw bounding boxes.

Usage:
    python3 enhance_sharpen.py [input_video] [output_video]

    Defaults:
        input_video: output_fixed.mp4
        output_video: processed_videos/sharpened_[timestamp].mp4
"""

import cv2
import numpy as np
import sys
import os
import datetime

# --- Tuning Parameters ---
# Kernel size: Larger = broader sharpening
BLUR_KERNEL_SIZE = (5, 5)
# Alpha: Weight for the original image
ALPHA = 1.5 
# Beta: Weight for the blurred image
BETA = -0.5 
# Gamma: Scalar added to each sum
GAMMA = 0 

def get_output_path(input_path, prefix="sharpened"):
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

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """
    Return a sharpened version of the image using an unsharp mask.
    amount determines the strength of the sharpening.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

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

        # Subtle Sharpening: amount=1.0. Increase for more aggressive sharpening, 
        # but be wary of increasing background artifact noise
        sharpened_frame = unsharp_mask(frame, kernel_size=BLUR_KERNEL_SIZE, amount=1.5, threshold=0)

        out.write(sharpened_frame)

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
        
    out_file = get_output_path(in_file, prefix="sharpened")
    process_video(in_file, out_file)
