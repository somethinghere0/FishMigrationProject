#!/usr/bin/env python3
"""
Interactive Exposure Tuner

This script allows you to interactively tune parameters for correcting overexposure
using an adaptive highlight mask. It isolates the brightest parts of the image and
darkens them, preserving shadows and mid-tones.

Usage:
    python3 tune_exposure.py [input_video]
    
    Default:
        input_video: output_fixed.mp4
        
Controls:
    - Press 'q' or ESC to quit
    - Move trackbars to adjust parameters
"""

import cv2
import numpy as np
import sys
import os

# Global variables for trackbars
threshold_val = 200    # Pixel intensity to consider "overexposed" (0-255)
reduction_val = 50     # Percentage to reduce brightness (0-100%)
blur_val = 21          # Kernel size for blurring the mask (must be odd)

frame = None

def get_odd_kernel(val):
    if val % 2 == 0:
        val += 1
    return max(3, val)

def correct_exposure(image, threshold, reduction_percent, blur_size):
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
    return np.clip(blended, 0, 255).astype(np.uint8), blurred_mask

def on_trackbar_change(val):
    update_image()

def update_image():
    global frame, threshold_val, reduction_val, blur_val

    if frame is None:
        return

    # Fetch current positions of trackbars
    try:
        threshold_val = cv2.getTrackbarPos('Highlight Threshold', 'Exposure Tuning')
        reduction_val = cv2.getTrackbarPos('Darkening %', 'Exposure Tuning')
        blur_val = cv2.getTrackbarPos('Mask Blur Size', 'Exposure Tuning')
    except cv2.error:
        pass

    actual_blur = get_odd_kernel(blur_val)
    
    # Apply correction
    corrected, mask = correct_exposure(
        frame, 
        threshold=threshold_val, 
        reduction_percent=reduction_val, 
        blur_size=actual_blur
    )
    
    # Create a split view showing the correction and the highlight mask
    # Resize mask to be a 3-channel small overlay
    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Show side by side or just toggle? Let's do a picture-in-picture for the mask
    h, w = corrected.shape[:2]
    pip_w, pip_h = int(w * 0.25), int(h * 0.25)
    mask_pip = cv2.resize(mask_display, (pip_w, pip_h))
    
    # Overlay PIP in bottom right
    padding = 10
    corrected[h - pip_h - padding : h - padding, 
              w - pip_w - padding : w - padding] = mask_pip
    
    # Add text overlay showing current parameters
    cv2.putText(corrected, f"Threshold: {threshold_val} (Lower = more areas affected)", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(corrected, f"Darkening: {reduction_val}% (Higher = darker highlights)", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(corrected, f"Blur Size: {actual_blur}", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(corrected, "Highlight Mask ->", (w - pip_w - 200, h - (pip_h//2)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
    cv2.imshow('Exposure Tuning', corrected)

def main():
    global frame, threshold_val, gamma_val, blur_val
    
    video_path = "output_fixed.mp4"
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
                
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        print("Usage: python3 tune_exposure.py [input_video.mp4]")
        sys.exit(1)

    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        sys.exit(1)

    # Find a frame that is bright
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = max(0, total_frames // 2)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame_read = cap.read()
    
    if not ret:
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

    cv2.namedWindow('Exposure Tuning', cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    
    # Create Trackbars
    cv2.createTrackbar('Highlight Threshold', 'Exposure Tuning', 0, 255, on_trackbar_change)
    cv2.createTrackbar('Darkening Strength (Gamma x10)', 'Exposure Tuning', 0, 50, on_trackbar_change)
    cv2.createTrackbar('Mask Blur Size', 'Exposure Tuning', 0, 100, on_trackbar_change)

    # Set initial positions (avoids macOS bug)
    cv2.setTrackbarPos('Highlight Threshold', 'Exposure Tuning', threshold_val)
    cv2.setTrackbarPos('Darkening Strength (Gamma x10)', 'Exposure Tuning', gamma_val)
    cv2.setTrackbarPos('Mask Blur Size', 'Exposure Tuning', blur_val)

    # Initial update
    update_image()

    print("\n--- Exposure Tuning Tool ---")
    print("Adjust trackbars to lower brightness on overexposed areas (like sandy bottoms).")
    print("TIP: Look at the small window in the corner to see what the script considers 'bright'.")
    print("Press 'q' or 'ESC' on the image window to close.")
    print("------------------------------\n")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
    
    final_gamma = gamma_val / 10.0
    final_blur = get_odd_kernel(blur_val)
    print(f"\nFinal Parameters:")
    print(f"THRESHOLD = {threshold_val}")
    print(f"GAMMA = {final_gamma}")
    print(f"BLUR_SIZE = {final_blur}")
    print("\nYou can update 'enhance_exposure.py' with these values.")

if __name__ == '__main__':
    main()
