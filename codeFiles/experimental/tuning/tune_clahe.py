#!/usr/bin/env python3
"""
Interactive CLAHE Tuner

This script allows you to interactively tune the parameters for CLAHE
(Contrast Limited Adaptive Histogram Equalization) on a given video frame.

Usage:
    python3 tune_clahe.py [input_video]
    
    Default:
        input_video: output_fixed.mp4
        
Controls:
    - Press 'q' or ESC to quit
    - Move trackbars to adjust CLAHE clip limit and tile grid size
"""

import cv2
import sys
import os

# Global variables for trackbars
clip_limit = 20  # Represents 2.0 (divided by 10)
tile_grid_size = 8
frame = None
original_lab = None

def on_clip_limit_change(val):
    global clip_limit
    clip_limit = max(1, val) # Prevent 0
    update_image()

def on_tile_grid_change(val):
    global tile_grid_size
    # Tile grid size must be >= 1. Usually powers of 2 (2, 4, 8, 16, 32) are best.
    tile_grid_size = max(1, val)
    update_image()

def update_image():
    global frame, original_lab, clip_limit, tile_grid_size

    if frame is None or original_lab is None:
        return

    # Real clip limit is trackbar value / 10
    actual_clip = clip_limit / 10.0
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=actual_clip, tileGridSize=(tile_grid_size, tile_grid_size))
    
    # Split LAB channels
    l, a, b = cv2.split(original_lab)
    
    # Apply CLAHE to L-channel
    l2 = clahe.apply(l)
    
    # Merge back and convert to BGR for display
    lab_merged = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
    
    # Add text overlay showing current parameters
    cv2.putText(enhanced, f"Clip Limit: {actual_clip:.1f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(enhanced, f"Tile Grid: {tile_grid_size}x{tile_grid_size}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
    cv2.imshow('CLAHE Tuning', enhanced)

def main():
    global frame, original_lab
    
    video_path = "output_fixed.mp4"
    if len(sys.argv) > 1:
        video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        print("Note: If you haven't run fix_barrel.py yet, run it to generate output_fixed.mp4")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        sys.exit(1)

    # Read the middle frame (where interesting things might be happening)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = max(0, total_frames // 2)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame_read = cap.read()
    
    if not ret:
        # Fallback to first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame_read = cap.read()
        if not ret:
            print("Error: Could not read any frame from the video.")
            sys.exit(1)

    cap.release()

    # Resize frame if it's too large to fit on screen
    h, w = frame_read.shape[:2]
    max_height = 800
    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        frame = cv2.resize(frame_read, (new_w, max_height))
    else:
        frame = frame_read.copy()
        
    # Pre-convert to LAB space for faster real-time updates
    original_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    cv2.namedWindow('CLAHE Tuning', cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    
    # Create Trackbars
    # Clip limit: 1 to 100 (represents 0.1 to 10.0)
    cv2.createTrackbar('Clip Limit (x10)', 'CLAHE Tuning', clip_limit, 100, on_clip_limit_change)
    # Tile Size: 1 to 32
    cv2.createTrackbar('Tile Size', 'CLAHE Tuning', tile_grid_size, 32, on_tile_grid_change)

    # Initial update
    update_image()

    print("\n--- CLAHE Tuning Tool ---")
    print("Adjust trackbars to find the best contrast enhancement.")
    print("Press 'q' or 'ESC' on the image window to close.")
    print("-------------------------\n")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
    
    final_clip = clip_limit / 10.0
    print(f"\nFinal Parameters:")
    print(f"clipLimit = {final_clip}")
    print(f"tileGridSize = ({tile_grid_size}, {tile_grid_size})")
    print("\nYou can update 'enhance_video.py' or create a new script with these values.")

if __name__ == '__main__':
    main()
