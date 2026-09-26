#!/usr/bin/env python3
"""
Test script for a different exposure correction method.
"""

import cv2
import numpy as np
import sys

def correct_exposure_v2(image, highlight_threshold=200, reduction_factor=0.5, blur_size=21):
    """
    Instead of a global LUT, this linearly scales down the brightness of pixels
    above the threshold.
    """
    image_float = image.astype(np.float32)
    
    # 1. Convert to grayscale to find bright spots
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Create mask of overexposed areas
    _, mask = cv2.threshold(gray, highlight_threshold, 255, cv2.THRESH_BINARY)
    
    # 3. Blur the mask for smooth blending
    blurred_mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    mask_norm = blurred_mask.astype(np.float32) / 255.0
    mask_norm_3c = cv2.merge([mask_norm, mask_norm, mask_norm])
    
    # 4. Create the darkened version
    # Simple linear reduction (e.g., multiply by 0.5 to cut brightness in half)
    darkened = image_float * reduction_factor
    
    # 5. Blend
    blended = image_float * (1.0 - mask_norm_3c) + darkened * mask_norm_3c
    return np.clip(blended, 0, 255).astype(np.uint8), blurred_mask

def main():
    video_path = "processed_videos/barrel_fixed_shortClip2_20260304_103739.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # Find a frame that is bright
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = max(0, total_frames // 2)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame_read = cap.read()
    
    if not ret:
        print("Failed to read frame")
        return
        
    corrected, mask = correct_exposure_v2(frame_read, highlight_threshold=200, reduction_factor=0.5, blur_size=21)
    
    cv2.imwrite("test_original.jpg", frame_read)
    cv2.imwrite("test_corrected.jpg", corrected)
    print("Saved test_original.jpg and test_corrected.jpg")

if __name__ == '__main__':
    main()
