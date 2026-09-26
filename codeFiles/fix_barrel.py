#!/usr/bin/env python3
"""
Standalone Barrel Distortion Correction Tool

This script corrects the specific fisheye/barrel distortion observed in the user's footage
using hardcoded parameters tuned for this camera.

Usage:
    python3 fix_barrel.py [input_video] [output_video]

    Defaults:
        input_video: testVideos/shortClip.mp4
        output_video: output_fixed.mp4
"""

import cv2
import numpy as np
import sys
import os

# --- Default Tuned Parameters ---
# Do not change unless retuning for a new lens/camera setup
PARAMS = {
    "f_scale": 1.362,
    "k1": -2.66,
    "k2": 4.03,
    "k3": -1.87,
    "k4": 2.71,
    "zoom": 1.12
}

def create_correction_maps(w, h, params):
    """
    Precomputes the remapping maps for fisheye correction.
    """
    f_scale = params['f_scale']
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']
    k4 = params['k4']
    zoom = params['zoom']
    
    # Intrinsic Matrix K
    # Assuming center at w/2, h/2
    K = np.array([[w * f_scale, 0, w/2],
                  [0, w * f_scale, h/2],
                  [0, 0, 1]], dtype=np.float32)
                  
    # Distortion Coefficients
    D = np.array([k1, k2, k3, k4], dtype=np.float32)
    
    # New Camera Matrix (with zoom)
    nk = K.copy()
    nk[0, 0] *= zoom
    nk[1, 1] *= zoom
    
    # Generate maps using OpenCV's fisheye model
    # Note: cv2.fisheye.initUndistortRectifyMap requires K, D, R, P, size, m1type
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, 
        np.eye(3), # R (Identity, no rotation) 
        nk,        # P (New Camera Matrix)
        (w, h), 
        cv2.CV_16SC2
    )
    
    return map1, map2

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
    
    # Generate maps once
    print("Generating correction maps...")
    map1, map2 = create_correction_maps(w, h, PARAMS)

    # Output setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"Processing... Output will be saved to '{output_path}'")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Geometric Correction ONLY
        corrected = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        out.write(corrected)

        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Processed {frame_count} frames ({progress:.1f}%)", end='\r')

    print(f"\nDone! Processed {frame_count} frames.")
    cap.release()
    out.release()

import datetime

def get_output_path(input_path, prefix="barrel_fixed"):
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
    
    new_filename = f"{prefix}_{name}_{timestamp}{ext}"
    return os.path.join(output_dir, new_filename)

if __name__ == "__main__":
    # Handle arguments
    in_file = "testVideos/shortClip.mp4"

    if len(sys.argv) > 1:
        in_file = sys.argv[1]

    # Optional second argument lets callers (e.g. the GUI pipeline) specify
    # an explicit output path; otherwise auto-generate a timestamped one.
    if len(sys.argv) > 2:
        out_file = sys.argv[2]
    else:
        out_file = get_output_path(in_file, prefix="barrel_fixed")

    process_video(in_file, out_file)
