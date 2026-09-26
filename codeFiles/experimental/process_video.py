#!/usr/bin/env python3
"""
Fisheye Video Corrector

This script applies the fisheye correction parameters found using
tune_fisheye.py to a full video file.

Usage:
    python3 process_video.py <input_video> <output_video> <params_json>
"""

import cv2
import numpy as np
import argparse
import json
import sys
import os

def load_params(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def process_video(input_path, output_path, params):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video {input_path}")
        sys.exit(1)
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Reconstruct the maps using the same logic as the tuner
    f_scale = params['f_scale']
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']
    k4 = params['k4']
    zoom = params['zoom']
    
    # Use the resolution from the video, which should match the params if tuned on the same source
    # If not, we scale K according to the ratio, but typically f_scale is relative to width anyway.
    w, h = width, height
    
    # Construct K
    K = np.array([[w * f_scale, 0, w/2],
                  [0, w * f_scale, h/2],
                  [0, 0, 1]], dtype=np.float32)
                  
    D = np.array([k1, k2, k3, k4], dtype=np.float32)
    
    # Construct New K with zoom
    nk = K.copy()
    nk[0, 0] *= zoom
    nk[1, 1] *= zoom
    
    # Compute maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), nk, (w, h), cv2.CV_16SC2)
    
    # Determine output size (same as input for now)
    out_size = (w, h)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, out_size)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    print(f"Processing {input_path} -> {output_path}")
    print(f"Resolution: {w}x{h}, FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        corrected = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # --- Contrast Enhancement (CLAHE) ---
        # Convert to LAB color space
        lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        l2 = clahe.apply(l)
        
        # Merge back and convert to BGR
        lab = cv2.merge((l2, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # --- Exposure Correction (Top-Right Corner) ---
        # Create a gradient mask for the top right corner
        # We want to darken it.
        rows, cols = enhanced.shape[:2]
        mask = np.ones((rows, cols), dtype=np.float32)
        
        # Define the region: Top 30% and Right 30%
        # Simple linear gradient from 1.0 down to 0.6 (40% darken) at the extreme corner
        center_x = cols
        center_y = 0
        radius = min(rows, cols) * 0.5  # Radius of influence
        
        # Access pixel coordinates grid
        y_grid, x_grid = np.indices((rows, cols))
        
        # Calculate distance from top-right corner
        # Optimization: precompute this outside loop if speed is issue, but for now explicit is fine
        dist = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        
        # Normalize distance: 0 at corner, 1 at radius
        # We only want to affect pixels within 'radius'
        influence = np.clip(1 - (dist / radius), 0, 1)
        
        # Darkening factor: 1.0 (no change) where influence is 0
        # 0.6 (darker) where influence is 1 (at the corner)
        adjustment = 1.0 - (influence * 0.4) 
        
        # Apply the mask
        # Expand dims for broadcasting over 3 channels
        adjustment = adjustment[:, :, np.newaxis]
        final = (enhanced.astype(np.float32) * adjustment).astype(np.uint8)
        
        out.write(final)
        
        count += 1
        if count % 30 == 0: # Print every 30 frames (approx 1 sec)
            print(f"Processed {count}/{total_frames} frames ({count/total_frames*100:.1f}%)", end='\r')
            
    cap.release()
    out.release()
    print(f"\nDone. Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Fisheye Video Processor')
    parser.add_argument('input_video', help='Input video file')
    parser.add_argument('output_video', help='Output video file')
    parser.add_argument('params_json', help='Parameters JSON file from tune_fisheye.py')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.params_json):
        print(f"Error: Params file {args.params_json} not found.")
        sys.exit(1)
        
    params = load_params(args.params_json)
    process_video(args.input_video, args.output_video, params)

if __name__ == '__main__':
    main()
