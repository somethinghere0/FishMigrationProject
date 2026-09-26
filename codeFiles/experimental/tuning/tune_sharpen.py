#!/usr/bin/env python3
"""
Interactive Sharpening Tuner

This script allows you to interactively tune the parameters for the Unsharp Mask
sharpening method on a given video frame. 

Because the video quality is poor, aggressive sharpening will amplify compression 
artifacts and noise. This tool helps you find the balance.

Usage:
    python3 tune_sharpen.py [input_video]
    
    Default:
        input_video: processed_videos/color_balanced_shortClip_20260217_205047_20260226_120428.mp4
        (or whatever your latest processed video is)
        
Controls:
    - Press 'q' or ESC to quit
    - Move trackbars to adjust sharpening amount, threshold, and kernel size
"""

import cv2
import numpy as np
import sys
import os

# Global variables for trackbars
amount_val = 10       # represents 1.0 (divided by 10)
threshold_val = 0     # 0 to 50
kernel_size_val = 5   # must be odd (3, 5, 7)

frame = None

def get_odd_kernel(val):
    if val % 2 == 0:
        val += 1
    return max(3, val)

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    
    if threshold > 0:
        low_contrast_mask = np.absolute(image.astype(np.int16) - blurred.astype(np.int16)) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def on_trackbar_change(val):
    update_image()

def update_image():
    global frame, amount_val, threshold_val, kernel_size_val

    if frame is None:
        return

    # Fetch current positions of trackbars
    try:
        amount_val = cv2.getTrackbarPos('Amount (x10)', 'Sharpening Tuning')
        kernel_size_val = cv2.getTrackbarPos('Kernel Size', 'Sharpening Tuning')
        threshold_val = cv2.getTrackbarPos('Threshold', 'Sharpening Tuning')
    except cv2.error:
        # Window might not be fully initialized yet
        pass

    # Real values
    actual_amount = amount_val / 10.0
    actual_kernel = get_odd_kernel(kernel_size_val)
    
    # Apply sharpening
    sharpened = unsharp_mask(
        frame, 
        kernel_size=(actual_kernel, actual_kernel), 
        amount=actual_amount, 
        threshold=threshold_val
    )
    
    # Add text overlay showing current parameters
    cv2.putText(sharpened, f"Amount: {actual_amount:.1f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(sharpened, f"Kernel: {actual_kernel}x{actual_kernel}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(sharpened, f"Threshold: {threshold_val}", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
    cv2.imshow('Sharpening Tuning', sharpened)

def main():
    global frame, amount_val, threshold_val, kernel_size_val
    
    # Try to find the most recently processed color balanced video to use as default if no arg provided
    video_path = None
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # User ran this right after enhance_color.py, look in processed_videos
        if os.path.exists("processed_videos"):
            videos = [os.path.join("processed_videos", f) for f in os.listdir("processed_videos") if f.startswith("color_balanced") and f.endswith(".mp4")]
            if videos:
                # Sort by modification time, newest first
                videos.sort(key=os.path.getmtime, reverse=True)
                video_path = videos[0]
                
    if not video_path or not os.path.exists(video_path):
        print(f"Error: Video file not found.")
        print("Usage: python3 tune_sharpen.py [input_video.mp4]")
        sys.exit(1)

    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        sys.exit(1)

    # Read the middle frame
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

    cv2.namedWindow('Sharpening Tuning', cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    
    # Create Trackbars (Name, Window, Initial, Max, Callback)
    # Amount: 0 to 50 (represents 0.0 to 5.0). Initialize to 0 to avoid macOS bug, set later.
    cv2.createTrackbar('Amount (x10)', 'Sharpening Tuning', 0, 50, on_trackbar_change)
    # Kernel Size: 3 to 15. Initialize to 0.
    cv2.createTrackbar('Kernel Size', 'Sharpening Tuning', 0, 15, on_trackbar_change)
    # Threshold: 0 to 50. Initialize to 0.
    cv2.createTrackbar('Threshold', 'Sharpening Tuning', 0, 50, on_trackbar_change)

    # Set initial positions
    cv2.setTrackbarPos('Amount (x10)', 'Sharpening Tuning', amount_val)
    cv2.setTrackbarPos('Kernel Size', 'Sharpening Tuning', kernel_size_val)
    cv2.setTrackbarPos('Threshold', 'Sharpening Tuning', threshold_val)

    # Initial update
    update_image()

    print("\n--- Sharpening Tuning Tool ---")
    print("Adjust trackbars to find the best sharpening balance.")
    print("TIP: Increase Threshold to avoid sharpening the background noise.")
    print("Press 'q' or 'ESC' on the image window to close.")
    print("------------------------------\n")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
    
    final_amount = amount_val / 10.0
    final_kernel = get_odd_kernel(kernel_size_val)
    print(f"\nFinal Parameters:")
    print(f"amount = {final_amount}")
    print(f"kernel_size = ({final_kernel}, {final_kernel})")
    print(f"threshold = {threshold_val}")
    print("\nYou can update 'enhance_sharpen.py' with these values.")

if __name__ == '__main__':
    main()
